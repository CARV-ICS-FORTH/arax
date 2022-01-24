#ifndef VTOP_VTWINDOW_HEADER_FILE
#define VTOP_VTWINDOW_HEADER_FILE

class VtopWindow {
    virtual void Draw()   = 0;
    virtual void Resize() = 0;
};
#endif
