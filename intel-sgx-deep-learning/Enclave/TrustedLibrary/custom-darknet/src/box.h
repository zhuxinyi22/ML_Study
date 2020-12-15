#ifndef BOX_H
#define BOX_H
#include "standard.h"

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
