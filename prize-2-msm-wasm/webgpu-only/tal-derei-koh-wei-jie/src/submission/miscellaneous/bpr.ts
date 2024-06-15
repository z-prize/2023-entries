import { ExtPointType } from "@noble/curves/abstract/edwards";

// Perform running sum in the classic fashion - one siumulated thread only
export const running_sum_bucket_reduction = (buckets: ExtPointType[]) => {
  const n = buckets.length;
  let m = buckets[0];
  let g = m;

  //console.log('<rs>')
  //console.log('\tm = buckets[0]; g = m')

  for (let i = 0; i < n - 1; i++) {
    const idx = n - 1 - i;
    //console.log(
    //`\tm = m.add(buckets[${idx}]);` +
    //`g = g.add(m)`
    //)
    const b = buckets[idx];
    m = m.add(b);
    g = g.add(m);
  }
  //console.log('</rs>')

  return g;
};

// Perform running sum with simulated parallelism. It is up to the caller
// to add the resulting points.
export const parallel_bucket_reduction = (
  buckets: ExtPointType[],
  num_threads = 4,
) => {
  const buckets_per_thread = buckets.length / num_threads;
  const bucket_sums: ExtPointType[] = [];

  //console.log('<parallel>')
  for (let thread_id = 0; thread_id < num_threads; thread_id++) {
    //console.log(`\t<thread ${thread_id}>`)

    // The thread ID
    const idx =
      thread_id === 0 ? 0 : (num_threads - thread_id) * buckets_per_thread;

    let m = buckets[idx];
    let g = m;
    //console.log(`\t\tm = buckets[${idx}]; g = m`)

    for (let i = 0; i < buckets_per_thread - 1; i++) {
      const idx = (num_threads - thread_id) * buckets_per_thread - 1 - i;
      //console.log(
      //`\t\tm = m.add(buckets[${idx}]); ` +
      //`g = g.add(m)`
      //)
      const b = buckets[idx];
      m = m.add(b);
      g = g.add(m);
    }

    // Perform scalar mul
    const s = buckets_per_thread * (num_threads - thread_id - 1);
    if (s > 0) {
      //console.log(`\t\tg.add(m ^ ${s})`)
      g = g.add(m.multiply(BigInt(s)));
    }

    bucket_sums.push(g);
    //console.log('\t</thread>')
  }
  //console.log('</parallel>')
  return bucket_sums;
};

// The first part of the parallel bucket reduction algo
export const parallel_bucket_reduction_1 = (
  buckets: ExtPointType[],
  num_threads = 4,
) => {
  const buckets_per_thread = buckets.length / num_threads;
  const g_points: ExtPointType[] = [];
  const m_points: ExtPointType[] = [];

  //console.log('<parallel>')
  for (let thread_id = 0; thread_id < num_threads; thread_id++) {
    //console.log(`\t<thread ${thread_id}>`)

    const idx =
      thread_id === 0 ? 0 : (num_threads - thread_id) * buckets_per_thread;

    let m = buckets[idx];
    let g = m;
    //console.log(`\t\tm = buckets[${idx}]; g = m`)

    for (let i = 0; i < buckets_per_thread - 1; i++) {
      const idx = (num_threads - thread_id) * buckets_per_thread - 1 - i;
      //console.log(
      //`\t\tm = m.add(buckets[${idx}]); ` +
      //`g = g.add(m)`
      //)
      const b = buckets[idx];
      m = m.add(b);
      g = g.add(m);
    }

    g_points.push(g);
    m_points.push(m);
    //console.log('\t</thread>')
  }
  //console.log('</parallel>')
  return { g_points, m_points };
};

// The second part of the parallel bucket reduction algo
export const parallel_bucket_reduction_2 = (
  g_points: ExtPointType[],
  m_points: ExtPointType[],
  num_buckets: number,
  num_threads = 4,
) => {
  const buckets_per_thread = num_buckets / num_threads;
  const result: ExtPointType[] = [];
  for (let thread_id = 0; thread_id < num_threads; thread_id++) {
    let g = g_points[thread_id];
    const m = m_points[thread_id];
    const s = buckets_per_thread * (num_threads - thread_id - 1);
    if (s > 0) {
      g = g.add(m.multiply(BigInt(s)));
    }
    result.push(g);
  }
  return result;
};
