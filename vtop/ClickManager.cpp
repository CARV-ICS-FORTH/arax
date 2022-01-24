#include "ClickManager.h"
#include <iostream>

std::vector<ClickManager :: ClickHandler *> ClickManager :: callbacks;

void ClickManager :: addClickHandler(ClickHandler *ch)
{
    callbacks.push_back(ch);
}

void ClickManager :: Click(int x, int y)
{
    for (auto & ch : ClickManager::callbacks) {
        if ( (x >= ch->sx) && (x < ch->ex) && (y >= ch->sy) && (y < ch->ey))
            ch->OnClick();
    }
}

void ClickManager :: Reset()
{
    callbacks.clear();
}
