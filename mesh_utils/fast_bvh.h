// fast_bvh.h – header-only BVH, SAH + object-median fallback
// Guarantees O(log N) depth. Drop-in for broken nanort SAH on real-world meshes.
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace fbvh
{

struct Ray
{
    float org[3]{};
    float dir[3]{};
    float min_t{0.f};
    float max_t{std::numeric_limits<float>::max()};
};

struct Hit
{
    float t{std::numeric_limits<float>::max()}; // P(t) = org + t * dir, org = ray origin, dir = ray direction
    float u{0.f}, v{0.f};
    uint32_t prim_id{~0u};

    bool valid() const
    {
        return prim_id != ~0u;
    }

    // World-space hit point via barycentric interpolation.
    // Usage: float p[3]; hit.position(bvh.verts_, bvh.faces_, p);
    void position(const float *verts, const uint32_t *faces, float out[3]) const
    {
        const uint32_t *tri = faces + prim_id * 3;
        const float *v0 = verts + tri[0] * 3;
        const float *v1 = verts + tri[1] * 3;
        const float *v2 = verts + tri[2] * 3;
        float w = 1.f - u - v;
        out[0] = w * v0[0] + u * v1[0] + v * v2[0];
        out[1] = w * v0[1] + u * v1[1] + v * v2[1];
        out[2] = w * v0[2] + u * v1[2] + v * v2[2];
    }

    // Flat (geometric) face normal – always available, no normal array needed.
    void face_normal(const float *verts, const uint32_t *faces, float out[3]) const
    {
        const uint32_t *tri = faces + prim_id * 3;
        const float *v0 = verts + tri[0] * 3;
        const float *v1 = verts + tri[1] * 3;
        const float *v2 = verts + tri[2] * 3;
        float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        out[0] = e1[1] * e2[2] - e1[2] * e2[1];
        out[1] = e1[2] * e2[0] - e1[0] * e2[2];
        out[2] = e1[0] * e2[1] - e1[1] * e2[0];
        float len = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
        if (len > 1e-12f)
        {
            out[0] /= len;
            out[1] /= len;
            out[2] /= len;
        }
    }

    // Smooth (interpolated) normal – requires per-vertex normals array (packed xyz).
    void smooth_normal(const float *normals, const uint32_t *faces, float out[3]) const
    {
        const uint32_t *tri = faces + prim_id * 3;
        const float *n0 = normals + tri[0] * 3;
        const float *n1 = normals + tri[1] * 3;
        const float *n2 = normals + tri[2] * 3;
        float w = 1.f - u - v;
        out[0] = w * n0[0] + u * n1[0] + v * n2[0];
        out[1] = w * n0[1] + u * n1[1] + v * n2[1];
        out[2] = w * n0[2] + u * n1[2] + v * n2[2];
        float len = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2]);
        if (len > 1e-12f)
        {
            out[0] /= len;
            out[1] /= len;
            out[2] /= len;
        }
    }
};

struct Node
{
    float bmin[3], bmax[3];
    uint32_t left_or_offset;
    uint32_t right_or_count;
    bool is_leaf;
};

struct PrimInfo
{
    float bmin[3], bmax[3], cen[3];
};

struct BVH
{
    std::vector<Node> nodes;
    std::vector<uint32_t> prim_ids;
    const float *verts_{};
    const uint32_t *faces_{};

    void build(const float *verts, const uint32_t *faces,
               uint32_t num_tris, uint32_t min_leaf = 16)
    {
        verts_ = verts;
        faces_ = faces;
        std::vector<PrimInfo> info(num_tris);
        for (uint32_t i = 0; i < num_tris; ++i)
        {
            const float *v0 = verts + faces[i * 3 + 0] * 3;
            const float *v1 = verts + faces[i * 3 + 1] * 3;
            const float *v2 = verts + faces[i * 3 + 2] * 3;
            for (int k = 0; k < 3; ++k)
            {
                float lo = std::min({v0[k], v1[k], v2[k]});
                float hi = std::max({v0[k], v1[k], v2[k]});
                info[i].bmin[k] = lo;
                info[i].bmax[k] = hi;
                info[i].cen[k] = (lo + hi) * .5f;
            }
        }
        prim_ids.resize(num_tris);
        for (uint32_t i = 0; i < num_tris; ++i)
            prim_ids[i] = i;
        nodes.clear();
        nodes.reserve(num_tris / std::max(1u, min_leaf / 2) + 4);
        build_r(info, 0, num_tris, min_leaf, 0);
    }

    bool traverse(const Ray &ray, Hit &hit) const
    {
        if (nodes.empty())
            return false;
        float inv[3];
        for (int k = 0; k < 3; ++k)
            inv[k] = std::abs(ray.dir[k]) > 1e-30f ? 1.f / ray.dir[k] : std::numeric_limits<float>::max();
        float t_best = ray.max_t;
        bool found = false;
        uint32_t stack[64];
        int sp = 0;
        stack[sp++] = 0;
        while (sp > 0)
        {
            const Node &n = nodes[stack[--sp]];
            if (!aabb_hit(ray.org, inv, n.bmin, n.bmax, ray.min_t, t_best))
                continue;
            if (n.is_leaf)
            {
                for (uint32_t j = 0; j < n.right_or_count; ++j)
                {
                    float t, u, v;
                    uint32_t pid = prim_ids[n.left_or_offset + j];
                    if (mt(ray, pid, t_best, t, u, v))
                    {
                        t_best = t;
                        hit = {t, u, v, pid};
                        found = true;
                    }
                }
            }
            else
            {
                stack[sp++] = n.right_or_count;
                stack[sp++] = n.left_or_offset;
            }
        }
        return found;
    }

  private:
    bool mt(const Ray &ray, uint32_t pid,
            float t_best, float &t, float &u, float &v) const
    {
        const uint32_t *tri = faces_ + pid * 3;
        const float *v0 = verts_ + tri[0] * 3, *v1 = verts_ + tri[1] * 3, *v2 = verts_ + tri[2] * 3;
        float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
        float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
        float h[3];
        cross(h, ray.dir, e2);
        float a = dot(e1, h);
        if (std::abs(a) < 1e-8f)
            return false;
        float f = 1.f / a;
        float s[3] = {ray.org[0] - v0[0], ray.org[1] - v0[1], ray.org[2] - v0[2]};
        u = f * dot(s, h);
        if (u < 0.f || u > 1.f)
            return false;
        float q[3];
        cross(q, s, e1);
        v = f * dot(ray.dir, q);
        if (v < 0.f || u + v > 1.f)
            return false;
        t = f * dot(e2, q);
        return t >= ray.min_t && t < t_best;
    }

    uint32_t build_r(const std::vector<PrimInfo> &info,
                     uint32_t begin, uint32_t end, uint32_t min_leaf, int depth)
    {
        uint32_t nid = (uint32_t)nodes.size();
        nodes.push_back({});
        float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
        for (uint32_t i = begin; i < end; ++i)
        {
            const PrimInfo &p = info[prim_ids[i]];
            for (int k = 0; k < 3; ++k)
            {
                bmin[k] = std::min(bmin[k], p.bmin[k]);
                bmax[k] = std::max(bmax[k], p.bmax[k]);
            }
        }
        for (int k = 0; k < 3; ++k)
        {
            nodes[nid].bmin[k] = bmin[k];
            nodes[nid].bmax[k] = bmax[k];
        }

        uint32_t cnt = end - begin;
        if (cnt <= min_leaf)
        {
            nodes[nid].is_leaf = true;
            nodes[nid].left_or_offset = begin;
            nodes[nid].right_or_count = cnt;
            return nid;
        }

        const int BINS = 32;
        float best_cost = std::numeric_limits<float>::max();
        int best_axis = -1;
        float best_split = 0.f;
        float node_sa = sa(bmin, bmax);
        for (int axis = 0; axis < 3; ++axis)
        {
            float lo = bmin[axis], hi = bmax[axis];
            if (hi - lo < 1e-9f)
                continue;
            float b_bmin[BINS][3], b_bmax[BINS][3];
            int b_cnt[BINS]{};
            for (int b = 0; b < BINS; ++b)
            {
                b_bmin[b][0] = b_bmin[b][1] = b_bmin[b][2] = 1e30f;
                b_bmax[b][0] = b_bmax[b][1] = b_bmax[b][2] = -1e30f;
            }
            float inv_r = BINS / (hi - lo);
            for (uint32_t i = begin; i < end; ++i)
            {
                const PrimInfo &p = info[prim_ids[i]];
                int b = std::max(0, std::min(BINS - 1, (int)((p.cen[axis] - lo) * inv_r)));
                ++b_cnt[b];
                for (int k = 0; k < 3; ++k)
                {
                    b_bmin[b][k] = std::min(b_bmin[b][k], p.bmin[k]);
                    b_bmax[b][k] = std::max(b_bmax[b][k], p.bmax[k]);
                }
            }
            float l_bmin[3] = {1e30f, 1e30f, 1e30f}, l_bmax[3] = {-1e30f, -1e30f, -1e30f};
            float lsa[BINS - 1];
            int lc[BINS - 1], acc = 0;
            for (int b = 0; b < BINS - 1; ++b)
            {
                acc += b_cnt[b];
                for (int k = 0; k < 3; ++k)
                {
                    l_bmin[k] = std::min(l_bmin[k], b_bmin[b][k]);
                    l_bmax[k] = std::max(l_bmax[k], b_bmax[b][k]);
                }
                lsa[b] = sa(l_bmin, l_bmax);
                lc[b] = acc;
            }
            float r_bmin[3] = {1e30f, 1e30f, 1e30f}, r_bmax[3] = {-1e30f, -1e30f, -1e30f};
            acc = 0;
            for (int b = BINS - 1; b > 0; --b)
            {
                acc += b_cnt[b];
                for (int k = 0; k < 3; ++k)
                {
                    r_bmin[k] = std::min(r_bmin[k], b_bmin[b][k]);
                    r_bmax[k] = std::max(r_bmax[k], b_bmax[b][k]);
                }
                if (!lc[b - 1] || !acc)
                    continue;
                float cost = 0.125f + (lsa[b - 1] * lc[b - 1] + sa(r_bmin, r_bmax) * acc) / node_sa;
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = lo + b * (hi - lo) / BINS;
                }
            }
        }
        if (best_axis < 0)
            best_axis = 0;
        auto part = std::partition(prim_ids.begin() + begin, prim_ids.begin() + end,
                                   [&](uint32_t p) { return info[p].cen[best_axis] < best_split; });
        uint32_t mid = (uint32_t)(part - prim_ids.begin());
        bool bad = (mid == begin || mid == end || (mid - begin) * 8 < cnt || (end - mid) * 8 < cnt);
        if (bad)
        {
            int ax = 0;
            float dx = bmax[0] - bmin[0], dy = bmax[1] - bmin[1], dz = bmax[2] - bmin[2];
            if (dy > dx)
                ax = 1;
            if (dz > ((ax == 0) ? dx : dy))
                ax = 2;
            mid = (begin + end) / 2;
            std::nth_element(prim_ids.begin() + begin, prim_ids.begin() + mid, prim_ids.begin() + end,
                             [&](uint32_t a, uint32_t b) { return info[a].cen[ax] < info[b].cen[ax]; });
        }
        mid = std::max(begin + 1, std::min(mid, end - 1));
        nodes[nid].is_leaf = false;
        uint32_t left = build_r(info, begin, mid, min_leaf, depth + 1);
        uint32_t right = build_r(info, mid, end, min_leaf, depth + 1);
        nodes[nid].left_or_offset = left;
        nodes[nid].right_or_count = right;
        return nid;
    }

    static float sa(const float *mn, const float *mx)
    {
        float dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
        return (dx < 0 || dy < 0 || dz < 0) ? 0.f : 2.f * (dx * dy + dy * dz + dz * dx);
    }
    static float dot(const float *a, const float *b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    static void cross(float *o, const float *a, const float *b)
    {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    }
    static bool aabb_hit(const float *org, const float *inv, const float *bmin, const float *bmax, float tmin, float tmax)
    {
        for (int k = 0; k < 3; ++k)
        {
            float t0 = (bmin[k] - org[k]) * inv[k], t1 = (bmax[k] - org[k]) * inv[k];
            if (inv[k] < 0.f)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);
            if (tmax < tmin)
                return false;
        }
        return true;
    }
};

inline BVH make_bvh(const float *verts, const uint32_t *faces, uint32_t n, uint32_t ml = 16)
{
    BVH b;
    b.build(verts, faces, n, ml);
    return b;
}

struct Stats
{
    int max_depth = 0;
    long leaves = 0, branches = 0;
};
inline Stats tree_stats(const BVH &bvh)
{
    Stats s;
    struct Frame
    {
        uint32_t nid;
        int d;
    };
    std::vector<Frame> stk;
    stk.push_back({0, 1});
    while (!stk.empty())
    {
        auto [nid, d] = stk.back();
        stk.pop_back();
        const Node &n = bvh.nodes[nid];
        if (n.is_leaf)
        {
            ++s.leaves;
            if (d > s.max_depth)
                s.max_depth = d;
        }
        else
        {
            ++s.branches;
            stk.push_back({n.left_or_offset, d + 1});
            stk.push_back({n.right_or_count, d + 1});
        }
    }
    return s;
}

} // namespace fbvh