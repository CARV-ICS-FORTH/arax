#ifndef CSV_HEADER_FILE
#define CSV_HEADER_FILE
#include <chrono>
#include <fstream>
#include <iostream>

class Csv {
public:
    Csv(const char *fname);
    Csv &print();
    std::chrono::time_point<std::chrono::system_clock> start;
    std::ofstream ofs;
};

template <class TYPE> Csv &operator << (Csv &csv, TYPE value)
{
    csv.ofs << " " << value;
    return csv;
}

#define CSV(FILE, EXPR) FILE.print() << EXPR << '\n'
#endif // ifndef CSV_HEADER_FILE
