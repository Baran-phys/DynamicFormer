import time

import numba

# from sympy import factorial2 as _factorial2
from sympy.abc import s
from scipy.special import zeta
import itertools
from numpy import linalg as LA
from fractions import Fraction
import numpy as np
import functools
from scipy.special import factorial2 as _factorial2
from functools import lru_cache

factorial = np.math.factorial
factorial2 = functools.partial(_factorial2, exact=True)

# def factorial2(x: int) -> int:
#     return _factorial2(x)  # type: ignore


@numba.jit(nopython=True)
def KroneckerDelta(x, y):
    if x == y:
        return 1.0
    return 0.0


# Initial data for Kontsevich volumes
@numba.jit(nopython=True)
def awk(i, j, k):
    if 0 == i == j == k:
        return 1
    return 0


def bwk(k, a, b):
    return (
        KroneckerDelta(k + a, b + 1)
        * factorial2(2 * b + 1)
        / (factorial2(2 * k + 1) * factorial2(2 * a + 1))
        * (2 * a + 1)
    )


def cwk(k, a, b):
    return KroneckerDelta(k, a + b + 2) * factorial2(2 * a + 1) * factorial2(2 * b + 1) / factorial2(2 * k + 1)


@numba.jit(nopython=True)
def dwk(k):
    if k == 1:
        return 1 / 24.0
    return 0.0


# Initial data for multicurve count
@numba.jit()
def amv(i, j, k):
    if 0 == i == j == k:
        return 1
    return 0


@numba.jit(nopython=False)
def bmv(k, a, b):
    return (2 * a + 1) * KroneckerDelta(k + a, b + 1) + KroneckerDelta(k, 0) * KroneckerDelta(a, 0) * zeta(
        2 * b + 2
    ) / s ** (2 * b + 2)


@numba.jit()
def cmv(k, a, b):
    p1 = 0
    if b - k + 1 >= 0:
        p1 = (
            factorial(2 * a + 2 * b - 2 * k + 3)
            / (factorial(2 * a + 1) * factorial(2 * b - 2 * k + 2))
            * zeta(2 * a + 2 * b - 2 * k + 4)
            / s ** (2 * a + 2 * b - 2 * k + 4)
        )

    p2 = 0
    if a - k + 1 >= 0:
        p2 = (
            factorial(2 * a + 2 * b - 2 * k + 3)
            / (factorial(2 * b + 1) * factorial(2 * a - 2 * k + 2))
            * zeta(2 * a + 2 * b - 2 * k + 4)
            / s ** (2 * a + 2 * b - 2 * k + 4)
        )

    return (
        KroneckerDelta(k, a + b + 2)
        + KroneckerDelta(k, 0) * KroneckerDelta(k, 0) * zeta(2 * a + 2) * zeta(2 * b + 2) / s ** (2 * a + 2 * b + 4)
        + p1
        + p2
    )


@numba.jit()
def dmv(k):
    return 1 / (2 * s**2) * zeta(2) * KroneckerDelta(k, 0) + 1 / 8 * KroneckerDelta(k, 1)


# Intermediate functions



# given a natural number n, this function return a list, whose element are the partitions of {2, ... , n}
def partition(lst):
    import more_itertools as mit
    return [part for k in range(1, len(lst) + 1) for part in mit.set_partitions(lst, k)]


# given a natural number n, this function return a list, whose element are the partitions of {2, ... , n}
# of the form {J_1, J_2}, with J_i nonempty
def bipartitions(n):
    collection = list(range(2, n + 1))
    #print(collection)
    return list(sorted(elem) for elem in partition(collection) if len(elem) == 2)


# given natural numbers h and n, this function return a list, whose element are the partitions of {2, ..., n} of
# the form {J_1, J_2}, with J_i nonempty if h=0.
def specialbipartitions(h, n):
    if h > 0:
        return bipartitions(n) + [[[], list(range(2, n))]]
    else:
        return bipartitions(n)


# given natural numbers n and d, this function return a list, whose element are the multiindices \[Mu]  \[Element]  \
# [DoubleStruckCapitalN]^n such that | \[Mu] | =  d
def multiindex(n, d):
    a = [range(0, d + 1)] * n
    return list(vector for vector in list(itertools.product(*a)) if LA.linalg.norm(vector, ord=1) <= d)


def noduplicate(myList):
    return sorted(set(myList))


@numba.jit(nopython=True)
def dim(g, n):
    return 3 * g - 3 + n


# **********************************************************************************************************************
# Topological recursion
@lru_cache(maxsize=None)  # Unlimited cache
def f(a, b, c, d, g, n, k):
    if (g == 0 and n == 1) or (g == 0 and n == 2) or g < 0 or n == 0:
        # print("f1: ")
        return 0
    elif g == 0 and n == 3:
        # print("f2: " )
        # print("k: "+str(k))
        # print("AWK: "+str(a(k[0], k[1], k[2])))
        return a(k[0], k[1], k[2])  # ************????
    elif g == 1 and n == 1:
        # print("f3: ")
        # print(k)
        # print("   k[0] :  "+str( k[0]) )
        # print("   d(k):  "+str( d(k[0]) ))
        return d(k[0])  # ************???
    else:
        # print("f4: ")
        # print("k -> fgn: " + str(k))
        return fgn(a, b, c, d, g, n, k)
        # simplify(expand(fgn(a, b, c, d, g, n, k)))  # ************


def drop(lis, notlis):
    notlis[:] = [number - 1 for number in notlis]
    reslisInx = set(range(len(lis))) - set(notlis)
    return [lis[i] for i in reslisInx]


# the amplitudes Subscript[F, g, n][k1 ... kn] for n>= 1
def fgn(a, b, c, d, g, n, k):
    r1 = 0
    if n > 1:
        for m in range(2, n + 1):
            for aa in range(0, dim(g, n - 1) - sum(drop(drop(k, [m]), [1])) + 1):
                # print("r1  k type  -->" + str(type(k)))
                # print("r1  [aa] + drop(drop(k, [m]), [1])  -->" + str([aa] + drop(drop(k, [m]), [1])))
                # print("r1  b  -->"+str(b(k[1-1], k[m-1], aa)))

                # print("r1  fgn  -->" + str(f(a, b, c, d, g, n - 1, [aa] + drop(drop(k, [m]), [1]))))
                r1 += b(k[1 - 1], k[m - 1], aa) * f(a, b, c, d, g, n - 1, [aa] + drop(drop(k, [m]), [1]))
    else:
        r1 = 0

    r2 = 0
    for aa in range(0, dim(g - 1, n + 1) - sum(k[1::]) + 1):
        for bb in range(0, dim(g - 1, n + 1) - sum(k[1::]) + 1):
            # print("r2  k  -->" + str(type(k)))
            # print("r2  k  -->" + str(k))
            # print("r2  k[1::]  -->" + str(k[1::]))
            # print("r2  c  -->" + str(c(k[1 - 1], aa, bb)))
            # print("r2  type(k[1::]) -->" + str(type(k[1::])))
            # print("r2  fgn  -->" + str((a, b, c, d, g - 1, n + 1, [aa] + [bb] + k[1::])))
            r2 += c(k[1 - 1], aa, bb) * f(a, b, c, d, g - 1, n + 1, [aa, bb] + k[1::])

    r3 = 0
    for aa in range(0, dim(g, n) + 1):
        for bb in range(0, dim(g, n) + 1):
            r31 = c(k[1 - 1], aa, bb)

            r32 = 0
            if n > 1:

                r321 = 0
                for hh in range(0, g + 1):
                    parts = specialbipartitions(hh, n)
                    # print("Parts: "+str(parts))
                    for part in parts:  # total1
                        kpart1 = [k[i - 1] for i in part[0]]
                        kpart2 = [k[i - 1] for i in part[1]]

                        # print("BiPart Prt 1:   "+str(len(part[1])))
                        # print("Original K->: " + str(k))
                        # print("Selected k: "+str([k[i-1] for i in part[1]]))
                        # print("Modified K->: "+str([bb] + [k[i-1] for i in part[1]]))
                        r321 += f(a, b, c, d, hh, 1 + len(kpart1), [aa] + kpart1) * f(
                            a, b, c, d, g - hh, 1 + len(kpart2), [bb] + kpart2
                        )

                r32 += r321

                r322 = 0
                for hh in range(0, g + 1):
                    parts = specialbipartitions(g - hh, n)
                    for part in parts:
                        kpart1 = [k[i - 1] for i in part[0]]
                        kpart2 = [k[i - 1] for i in part[1]]
                        r322 += f(a, b, c, d, hh, 1 + len(kpart2), [aa] + kpart2) * f(
                            a, b, c, d, g - hh, 1 + len(kpart1), [bb] + kpart1
                        )
                        
                r32 += r322

            else:

                r32 = 0
                for hh in range(1, g - 1 + 1):
                    r32 = f(a, b, c, d, hh, 1, [aa]) * f(a, b, c, d, g - hh, 1, [bb]) + r32

            r3 = r31 * r32
    # print("r1:   "+str(r1))
    # print("r2:   " + str(r2))
    # print("r3:  " + str(r3))
    # print("FGN Result:  " + str(r1 + 1 / 2 * r2 + 1 / 2 * r3))
    return r1 + 1 / 2 * r2 + 1 / 2 * r3


# ***********************************************************************************************************************
# Better output
# function to convert to subscript
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans("".join(normal), "".join(sub_s))
    return x.translate(res)


def kontVolume(g, n):
    # print("multiindex")
    # print(multiindex(n, dim(g, n)))
    coefList = []
    for k in multiindex(n, dim(g, n)):
        k = list(k)
        p = 1
        for i in range(n):
            p *= 1 / (factorial(k[i - 1]) * 2 ** k[i - 1])
        if p * f(awk, bwk, cwk, dwk, g, n, k) != 0:
            c = Fraction(str(p * f(awk, bwk, cwk, dwk, g, n, k))).limit_denominator(1000000)
        else:
            c = 0
        coefList += [c]

    print("V{},{}(L)={}".format(get_sub(str(g)), get_sub(str(n)), str(coefList)))


def amp(g, n):
    print(f"The number of partitions are {len(multiindex(n, dim(g, n)))}")
    print(multiindex(n, dim(g, n)))
    fgn_list = []
    for k in multiindex(n, dim(g, n)):
        k = list(k)
        # k = np.array(k)
        print(k)
        # if f(awk, bwk, cwk, dwk, g, n,  k) !=0 :
        fgn = f(awk, bwk, cwk, dwk, g, n, k)
        fgn_list.append(fgn)
    print(fgn_list)


start_time = time.time()
amp(2, 2)
print("--- %s seconds ---" % (time.time() - start_time))

# kontVolume(0, 3)
# kontVolume(1, 1)
# kontVolume(0, 4)
# kontVolume(1, 2)
