import math

import numpy as np
import itertools
from qiskit.quantum_info import Pauli, PauliList
from typing import List, Tuple, Dict, Optional, Set, Union
import sys
from numpy.linalg import matrix_rank
from numpy.linalg import matrix_power as m_power

from qiskit.quantum_info import Statevector

def bring_states():
    state_list= [0, 0, 0, 0, 0, 0,
                 0, -1/(2*np.sqrt(2))*1j, 1/(2*np.sqrt(2))*1j,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,
                 0, -1/(2*np.sqrt(2))*1j, 0,0, 0, 0,
                 0, 0, 1/(2*np.sqrt(2))*1j,0, 0, 0,
                 0, 0, 0,0, 0, 0, 0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, -1/(2*np.sqrt(2))*1j, 0,0, 0, 0,
                 0, 0, 0,0, 0, 0,
                 1/(2*np.sqrt(2))*1j, 0, 0,0, -1/(2*np.sqrt(2))*1j, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 1/(2*np.sqrt(2))*1j,0, 0, 0,
                 0, 0, 0,0, 0, 0,0, 0, 0,
                 0, 0, 0,0, 0]
    State = Statevector(state_list)
    return State


def hamming_distance(s1: Union[str, List[int], Tuple[int]],
                     s2: Union[str, List[int], Tuple[int]]):
    distance = 0
    for i in range(len(s1)):
        # Convert characters to integers if input is string
        bit1 = int(s1[i]) if isinstance(s1, str) else s1[i]
        bit2 = int(s2[i]) if isinstance(s2, str) else s2[i]
        if bit1 != bit2:
            distance += 1
    return distance


def minimum_distance(code: List[Union[str, List[int], Tuple[int]]]) -> int:
    """
    Calculates the minimum Hamming distance for a given code.

    Args:
        code: A list where each element is a codeword
              (string, list, or tuple of 0s and 1s).
              Assumes all codewords have the same length.

    Returns:
        The minimum Hamming distance (d) of the code.
        Returns float('inf') if the code has less than 2 codewords.
    """
    num_codewords = len(code)
    if num_codewords < 2:
        # Minimum distance is not well-defined or is infinite
        return float('inf')

    # Assuming all codewords have the same length as the first one
    codeword_length = len(code[0])
    min_dist = codeword_length + 1 # Initialize with a value larger than any possible distance

    for i in range(num_codewords):
        for j in range(i + 1, num_codewords):
            dist = hamming_distance(code[i], code[j])
            if dist < min_dist:
                min_dist = dist

    return min_dist


# matrix rank over GF(2) 
def matrixRank(mat):
    M=mat.copy() # mat should be an np.array
    m=len(M) # number of rows
    pivots={} # dictionary mapping pivot row --> pivot column
    # row reduction
    for row in range(m):
        pos = next((index for index,value in enumerate(M[row]) if value != 0), -1) #finds position of first nonzero element (or -1 if all 0s)
        if pos>-1:
            for row2 in range(m):
                if row2!=row and M[row2][pos]==1:
                    M[row2]=((M[row2]+M[row]) % 2)
            pivots[row]=pos
    return len(pivots)


