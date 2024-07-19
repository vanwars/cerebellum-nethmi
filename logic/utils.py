def show_digit(label, bitmap, sigfigs=3, convert_to_int=False, hide_zeros=True):
    num_cols = 28 * (sigfigs + 1)
    print(num_cols * '-')
    print("This is a", label)
    print(num_cols * '-')
    for row in bitmap:
        for pixel in row:
            if convert_to_int:
                pixel = int(pixel)
            if hide_zeros and pixel == 0:
                pixel = ""
            print(f'{pixel:>{sigfigs}}', end="|")
        print("")


def show_first_n_digits(training_bitmaps, training_labels, n=5, sigfigs=3, convert_to_int=False, hide_zeros=True):
    i = 0
    for label in training_labels:
        if i > n:
            break
        show_digit(
            label, training_bitmaps[i], sigfigs=sigfigs, convert_to_int=convert_to_int, hide_zeros=hide_zeros)
        i += 1

