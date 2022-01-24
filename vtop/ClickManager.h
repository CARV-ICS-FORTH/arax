#ifndef VTOP_CLICK_MANAGER_HEADER_FILE
#define VTOP_CLICK_MANAGER_HEADER_FILE
#include <vector>

class ClickManager
{
public:
    struct ClickHandler
    {
        virtual void OnClick() = 0;
        int sx;
        int sy;
        int ex;
        int ey;
    };
    static void addClickHandler(ClickHandler *ch);
    static void Click(int x, int y);
    static void Reset();
private:
    struct CbPair
    {
        void  (*fn)(void *);
        void *dt;
    };
    static std::vector<ClickHandler *> callbacks;
};
#endif // ifndef VTOP_CLICK_MANAGER_HEADER_FILE
