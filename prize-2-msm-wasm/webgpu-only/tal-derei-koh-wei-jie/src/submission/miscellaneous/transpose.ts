/// Typescript implementation of serial transpose algorithm from
/// https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf.
/// It simulates running multiple transpositions in parallel, with one thread
/// per CSR matrix. It does not accept an arbitrary csr_row_ptr array.

const calc_start_end = (m: number, n: number, i: number) => {
  if (i < m) {
    return [i * n, i * n + n];
  } else {
    return [m * n, m * n];
  }
};

export const cpu_transpose = (
  all_csr_col_idx: number[],
  n: number, // The width of the matrix
  m: number, // The height of the matrix
  num_subtasks: number,
  input_size: number,
) => {
  const all_csc_col_ptr = Array(num_subtasks * (n + 1)).fill(0);
  const all_csc_row_idx = Array(num_subtasks * input_size).fill(0);
  const all_csc_vals = Array(num_subtasks * input_size).fill(0);
  const all_curr = Array(num_subtasks * n).fill(0);

  for (let subtask_idx = 0; subtask_idx < num_subtasks; subtask_idx++) {
    const ccp_offset = subtask_idx * (n + 1);
    const cci_offset = subtask_idx * input_size;
    const curr_offset = subtask_idx * n;

    // Calculate the count per column. This step is *not* parallelisable because
    // the index `csr_col_idx[j] + 1` can be the same across iterations,
    // causing a race condition.
    for (let i = 0; i < m; i++) {
      const [start, end] = calc_start_end(m, n, i);
      for (let j = start; j < end; j++) {
        all_csc_col_ptr[ccp_offset + all_csr_col_idx[cci_offset + j] + 1]++;
      }
    }

    // Prefix sum, aka cumulative/incremental sum
    for (let i = 1; i < n + 1; i++) {
      all_csc_col_ptr[ccp_offset + i] += all_csc_col_ptr[ccp_offset + i - 1];
    }

    let val = 0;
    for (let i = 0; i < m; i++) {
      const [start, end] = calc_start_end(m, n, i);
      for (let j = start; j < end; j++) {
        const loc =
          all_csc_col_ptr[ccp_offset + all_csr_col_idx[cci_offset + j]] +
          all_curr[curr_offset + all_csr_col_idx[cci_offset + j]]++;

        all_csc_row_idx[subtask_idx * input_size + loc] = i;
        all_csc_vals[cci_offset + loc] = val;
        val++;
      }
    }
  }

  return { all_csc_col_ptr, all_csc_row_idx, all_csc_vals };
};

/**
 * Serial transpose algo from
 * https://synergy.cs.vt.edu/pubs/papers/wang-transposition-ics16.pdf
 */
export const cpu_transpose_classic = (
  csr_row_ptr: number[],
  csr_col_idx: number[],
  n: number, // The width of the matrix
) => {
  // The height of the matrix
  const m = csr_row_ptr.length - 1;

  const curr: number[] = Array(n).fill(0);
  const csc_col_ptr: number[] = Array(n + 1).fill(0);
  const csc_row_idx: number[] = Array(csr_col_idx.length).fill(0);
  const csc_vals: number[] = Array(csr_col_idx.length).fill(0);

  // Calculate the count per column. This step is *not* parallelisable because
  // the index `csr_col_idx[j] + 1` can be the same across iterations,
  // causing a race condition.
  for (let i = 0; i < m; i++) {
    const start = csr_row_ptr[i];
    const end = csr_row_ptr[i + 1];
    for (let j = start; j < end; j++) {
      csc_col_ptr[csr_col_idx[j] + 1]++;
    }
  }

  // Prefix sum, aka cumulative/incremental sum
  for (let i = 1; i < n + 1; i++) {
    csc_col_ptr[i] += csc_col_ptr[i - 1];
  }

  let val = 0;
  for (let i = 0; i < m; i++) {
    const start = csr_row_ptr[i];
    const end = csr_row_ptr[i + 1];
    for (let j = start; j < end; j++) {
      const loc = csc_col_ptr[csr_col_idx[j]] + curr[csr_col_idx[j]]++;
      csc_row_idx[loc] = i;
      csc_vals[loc] = val;
      val++;
    }
  }

  return { csc_col_ptr, csc_row_idx, csc_vals };
};
