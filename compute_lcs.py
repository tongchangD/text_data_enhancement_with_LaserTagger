# coding=utf-8
# 函数compute_lcs，用于计算两个列表的Longest Common Subsequence (LCS)

def _compute_lcs(source, target):
  """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    source: List of source tokens.
    target: List of target tokens.

  Returns:
    List of tokens in the LCS.
  """
  table = _lcs_table(source, target)
  return _backtrack(table, source, target, len(source), len(target))


def _lcs_table(source, target):
  """Returns the Longest Common Subsequence dynamic programming table."""
  rows = len(source)
  cols = len(target)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if source[i - 1] == target[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def _backtrack(table, source, target, i, j):
  """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
    table: Precomputed LCS table.
    source: List of source tokens.
    target: List of target tokens.
    i: Current row index.
    j: Current column index.

  Returns:
    List of tokens corresponding to LCS.
  """
  if i == 0 or j == 0:
    return []
  if source[i - 1] == target[j - 1]:
    # Append the aligned token to output.
    return _backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
  if table[i][j - 1] > table[i - 1][j]:
    return _backtrack(table, source, target, i, j - 1)
  else:
    return _backtrack(table, source, target, i - 1, j)

