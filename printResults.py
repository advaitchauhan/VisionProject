import pickle
with open('mnist_results.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    n_i0, zn_i0, n_i1, zn_i1, n_im, zn_im = pickle.load(f)
    print("Simple Init, Noise")
    print(*n_i0)
    print("Simple Init, No Noise ")
    print(*zn_i0)
    print("Good Init, Noise")
    print(*n_i1)
    print("Good Init, No Noise")
    print(*zn_i1)
    print("My Init, Noise")
    print(*n_im)
    print("My Init, No Noise")
    print(*zn_im)