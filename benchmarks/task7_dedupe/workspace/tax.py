def calc_with_tax_us(price):
    if price < 0:
        raise ValueError("price must be non-negative")
    return round(price * 1.075, 2)


def calc_with_tax_eu(price):
    if price < 0:
        raise ValueError("price must be non-negative")
    return round(price * 1.20, 2)


def calc_with_tax_jp(price):
    if price < 0:
        raise ValueError("price must be non-negative")
    return round(price * 1.10, 2)


def calc_with_tax_uk(price):
    if price < 0:
        raise ValueError("price must be non-negative")
    return round(price * 1.20, 2)
