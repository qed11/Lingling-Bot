def draw_hilbert(dim, startx = 0, starty = 0, pos_step = True, xfirst = False):
    if dim == 2:
        if pos_step:
            if not xfirst:
                return [(startx, starty), (startx, starty+1), (startx+1, starty+1), (startx+1, starty)]
            else:
                return [(startx, starty), (startx+1, starty), (startx+1, starty+1), (startx, starty+1)]
        else:
            if not xfirst:
                return [(startx, starty), (startx, starty-1), (startx-1, starty-1), (startx-1, starty)]
            else:
                return [(startx, starty), (startx-1, starty), (startx-1, starty-1), (startx, starty-1)]
    else:
        sign = 1
        if not pos_step:
            sign = -1
        if not xfirst:
            startx2, starty2 = startx, starty+dim*sign/2
            startx3, starty3 = startx2+dim*sign/2, starty2
            startx4, starty4 = startx3+(dim/2-1)*sign, starty3-sign
        else:
            startx2, starty2 = startx+dim*sign/2, starty
            startx3, starty3 = startx2, starty2+dim*sign/2
            startx4, starty4 = startx3-sign, starty3+(dim/2-1)*sign
        return draw_hilbert(dim/2, startx, starty, pos_step, not xfirst) + draw_hilbert(dim/2, startx2, starty2, pos_step, xfirst) + draw_hilbert(dim/2, startx3, starty3, pos_step, xfirst) + draw_hilbert(dim/2, startx4, starty4, not pos_step, not xfirst)
