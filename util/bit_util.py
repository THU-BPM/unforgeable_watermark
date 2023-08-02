

def int_to_bin_list(n, length=8):
    bin_str = format(n, 'b').zfill(length)
    return [int(b) for b in bin_str]