#include "read_ply.h"


Mesh load_ply(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open: " + path);
    bool binary_le = false;
    int nv = 0, nf = 0, xi = -1, yi = -1, zi = -1, pc = 0;
    bool inv = false;
    std::string line;
    while (std::getline(f, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        std::istringstream ss(line);
        std::string t;
        ss >> t;
        if (t == "format")
        {
            std::string fmt;
            ss >> fmt;
            binary_le = (fmt == "binary_little_endian");
        }
        else if (t == "element")
        {
            std::string nm;
            long c;
            ss >> nm >> c;
            if (nm == "vertex")
            {
                nv = (int)c;
                inv = true;
                pc = 0;
            }
            else
            {
                nf = (int)c;
                inv = false;
            }
        }
        else if (t == "property")
        {
            std::string tp;
            ss >> tp;
            if (tp != "list" && inv)
            {
                std::string nm;
                ss >> nm;
                if (nm == "x")
                    xi = pc;
                if (nm == "y")
                    yi = pc;
                if (nm == "z")
                    zi = pc;
                ++pc;
            }
        }
        else if (t == "end_header")
            break;
    }
    if (xi < 0 || yi < 0 || zi < 0)
        throw std::runtime_error("PLY: no xyz");
    Mesh m;
    m.verts.resize(nv * 3);
    if (binary_le)
    {
        std::vector<float> row(pc);
        for (int i = 0; i < nv; ++i)
        {
            f.read((char *)row.data(), pc * 4);
            m.verts[i * 3] = row[xi];
            m.verts[i * 3 + 1] = row[yi];
            m.verts[i * 3 + 2] = row[zi];
        }
        for (int i = 0; i < nf; ++i)
        {
            uint8_t c8;
            f.read((char *)&c8, 1);
            std::vector<uint32_t> idx(c8);
            f.read((char *)idx.data(), c8 * 4);
            for (int j = 1; j + 1 < (int)c8; ++j)
            {
                m.faces.push_back(idx[0]);
                m.faces.push_back(idx[j]);
                m.faces.push_back(idx[j + 1]);
            }
        }
    }
    else
    {
        for (int i = 0; i < nv; ++i)
        {
            std::string ln;
            std::getline(f, ln);
            std::istringstream ss2(ln);
            std::vector<float> vals;
            float v;
            while (ss2 >> v)
                vals.push_back(v);
            m.verts[i * 3] = vals[xi];
            m.verts[i * 3 + 1] = vals[yi];
            m.verts[i * 3 + 2] = vals[zi];
        }
        for (int i = 0; i < nf; ++i)
        {
            std::string ln;
            std::getline(f, ln);
            std::istringstream ss2(ln);
            int cnt;
            ss2 >> cnt;
            std::vector<uint32_t> idx(cnt);
            for (int j = 0; j < cnt; ++j)
                ss2 >> idx[j];
            for (int j = 1; j + 1 < cnt; ++j)
            {
                m.faces.push_back(idx[0]);
                m.faces.push_back(idx[j]);
                m.faces.push_back(idx[j + 1]);
            }
        }
    }
    return m;
}