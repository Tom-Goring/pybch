import numpy as np
from sympy import Matrix
from mod import Mod

np.set_printoptions(suppress=True)


class NoValidRoots(Exception):
    pass


class BCH(object):
    """ A class to hold information about a BCH code

     Args:
         codeword_length (int): the length of the encoded message (n)
         message_length (int): the length of the message to be encoded (k)
     """

    def __init__(self, codeword_length: int, message_length: int):
        self.codeword_length = codeword_length
        self.message_length = message_length

        gf = codeword_length + 1
        self.GF = gf

        m = codeword_length - message_length  # m = number of parity bits
        self.M = m

        p = np.array(range(1, gf))  # polynomial sequence to be made into vandermonde matrix
        vandermonde = np.rot90(np.column_stack([p ** (m - 1 - i) % gf for i in range(m)]))
        sub_matrix = vandermonde[:, message_length:]  # we take the end of the matrix

        m = Matrix(sub_matrix)
        inv = np.array(m.inv_mod(gf))  # and create its modular inverse

        # find the dot product of the inverse and the vandermonde matrix (this is the same as doing arbitrary row
        # operations until you have an identity matrix in the same sub-matrix
        gen = (np.mod(np.dot(inv, vandermonde), gf)[:, :message_length] * -1) % gf

        # create and glue a big identity matrix to our generator to allow encoding to just be a simple dot product
        ident = np.identity(message_length)

        self.G = np.hstack((ident, np.rot90(np.fliplr(gen)))).astype(int)

    def encode(self, i: np.array):
        """ Returns the BCH codeword for the given input """
        return np.rint(np.mod(np.dot(i, self.G), self.GF)).astype(int)

    def decode(self, code: np.array):
        """ Returns the corrected (if errors have been detected) message corresponding to the given BCH codeword """
        p = np.array(range(1, self.GF))
        parity_check_matrix = np.column_stack([p ** (self.M - 1 - i) for i in range(self.M)])
        s = np.mod(np.dot(code, parity_check_matrix), self.GF)[::-1]

        if all([e == 0 for e in s]):
            return code
        else:
            s = [Mod(e, self.GF) for e in s]
            (p, q, r) = (
                (s[1] ** 2) - (s[0] * s[2]),
                (s[0] * s[3]) - (s[1] * s[2]),
                (s[2] ** 2) - (s[1] * s[3])
            )

            if (p, q, r) == (0, 0, 0):
                a = v(s[0])
                i = v((s[1] // a))
                code[i - 1] = (code[i - 1] - a) % self.GF
                return list(code), (i, a)
            else:
                try:
                    i = (-q + sqrt(q ** 2 - 4 * p * r)) // (2 * p)
                    j = (-q - sqrt(q ** 2 - 4 * p * r)) // (2 * p)

                    if i == 0 or j == 0:
                        return None

                    b = v((i * s[0] - s[1]) // (i - j))
                    a = v(s[0] - b)

                    (i, j) = (v(i), v(j))  # convert from mod type to int to index array

                    code[i - 1] = (code[i - 1] - a) % self.GF
                    code[j - 1] = (code[j - 1] - b) % self.GF

                    if code[i - 1] == self.GF - 1 or code[j - 1] == self.GF - 1:
                        return None

                    return list(code), (i, a), (j, b)
                except NoValidRoots:
                    return None
                except ValueError:
                    return None


def sqrt(n: Mod):
    for i in range(n.modulus):
        if i ** 2 % n.modulus == n:
            return i
    raise NoValidRoots


# noinspection PyProtectedMember
def v(n: Mod):
    return n._value


if __name__ == '__main__':
    bch = BCH(16, 12)
    # num = [int(x) for x in str(1145195876)]
    print(bch.decode([4, 2, 1, 2, 7, 5, 6, 5, 8, 9, 11, 0, 6, 3, 3, 12]))
