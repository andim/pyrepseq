import numpy as np
import subprocess
from io import StringIO
from Bio import SeqIO
import logomaker as lm
from pandas import DataFrame
from warnings import warn


def align_seqs(seqs, debug=False):
    """Align multiple sequences using mafft-linsi with default parameters.

    Requires external dependency mafft-linsi to be installed.

    Parameters
    ----------
    seqs: iterable of strings
    debug: Boolean
        if True, prints mafft-linsi output

    Returns
    -------
    list of strings
        aligned sequences (with gaps)
    """
    seq_str = ""
    for i, seq in enumerate(seqs):
        seq_str += f"> seq {i}\n"
        seq_str += f"{seq}\n"
    if debug:
        child = subprocess.Popen(
            ["mafft-linsi", "--amino", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
    else:
        child = subprocess.Popen(
            ["mafft-linsi", '--quiet', "--amino", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
    child.stdin.write(seq_str.encode())
    child_out = child.communicate()[0].decode("utf8")
    seqs_aligned = list(SeqIO.parse(StringIO(child_out), "fasta"))
    child.stdin.close()
    return [str(seq.seq) for seq in seqs_aligned]


def seqs_to_regex(seqs, align=True):
    """Turn a list of sequences into a regular expression.

    The regular expression matches sequences to what is effectively an independent site model.
    The model assumes equal frequencies for all observed residues/gaps.
    """
    if align:
        seqs = align_seqs(seqs)
    matrix = lm.alignment_to_matrix(seqs)
    n = len(seqs)
    regex = ''
    for i, row in matrix.iterrows():
        s = ''.join(row[row>0].index)
        if len(s)>1:
                regex += f'[{s}]'
        else:
            regex += s
        gaps = row.sum()!=n
        if gaps:
            regex += '?'
    return regex
    

def seqs_to_consensus(seqs, align=True):
    """Turn a list of sequences into a consensus sequence.

    The consensus sequence consists of the most frequent amino acid at each site.

    align: boolean
        if False all sequences need to be of the same length
        if True requires mafft-linsi to be installed
    """
    if align:
        seqs = align_seqs(seqs)
    matrix = lm.alignment_to_matrix(seqs)
    n = len(seqs)
    s = ''
    for i, row in matrix.iterrows():
        ngaps = n-row.sum()
        if ngaps > n//2:
            continue
        s += row.idxmax()
    return s


def ensure_numpy(arr_like):
    module = type(arr_like).__module__
    if module == "pandas.core.series":
        return arr_like.to_numpy()
    if module == "numpy":
        return arr_like
    return np.array(arr_like)


def convert_tuple_to_dataframe_if_necessary(seqs):
    if not (isinstance(seqs, tuple) and len(seqs) == 2):
        return seqs

    warn("Inputting paired-chain CDR3 data as a tuple of Iterable[str]s is now deprecated. Please use the standard pyrepseq TCR DataFrame format instead (https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.io.standardize_dataframe).")
    seqs_rowwise = zip(*seqs)
    return DataFrame(data=seqs_rowwise, columns=("CDR3A", "CDR3B"))
