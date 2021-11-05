

def eventX(t, s, value, mc):
    return s[0] - value


def eventY(t, s, value, mc):
    return s[1] - value


def eventSecondaryRdotV(t, s, value, mc):
    r = s[:3].copy()
    r[0] -= 0.9999969986516103
    v = s[3:6]
    return r @ v
