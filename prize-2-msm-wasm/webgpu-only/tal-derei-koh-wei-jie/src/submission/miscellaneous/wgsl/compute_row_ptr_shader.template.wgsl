// Input buffers
@group(0) @binding(0)
var<storage, read> new_point_indices: array<u32, {{ num_chunks }}>;

// Output buffers
@group(0) @binding(1)
var<storage, read_write> row_ptr: array<u32>;

@compute
@workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // col_idx is the scalar chunk associated with each point
    // row_ptr is the running sum of points per row
    // inputs:
    //   - new_points (x, y, t, z)
    //   - scalar_chunks
    //   - new_point_indices
    //   - input_size
    //   - num_subtasks
    //
    // Each row has a maximum of num_chunks / num_subtasks / num_rows points
    // Example 1:
    //   - new_points: [P0, P1, P2, P3]
    //   - chunks: [4, 8, 12, 16]
    //   - new_point_indices: [0, 1, 2, 3]
    //   - num_rows = 2
    //   - max_row_size = 2
    //   - row_ptr: [0, 2, 4]
    //
    // Example 2:
    //   - new_points: [P0, P1, P2]
    //   - chunks: [4, 8, 4, 0]
    //   - new_point_indices: [0, 2, 1, 0]
    //   - num_rows = 2
    //   - max_row_size = 2
    //   - row_ptr: [0, 2, 3]
    //
    // Example 3:
    //   - new_points: [P0, P1, P2, P3, P4, P5, P6, P7]
    //   - chunks: [4, 8, 4, 1, 2, 3, 5, 0]
    //   - new_point_indices: [0, 2, 1, 3, 4, 5, 6, 0]
    //   - num_chunks = 8
    //   - num_rows = 4
    //   - max_row_size = 8 / 4 = 2
    //   - row_ptr = [0, 2, 4, 6, 7]
    //   
    // Assuming that new_point_indices always starts with 0, the shader
    // needs to stop at the first 0 value whose index is greater than 0.
    let max_row_size = {{ max_row_size }}u;
    let num_chunks = {{ num_chunks }}u;

    row_ptr[0] = 0u;
    var k = 1u;
    for (var i = 0u; i < num_chunks; i += max_row_size) {
       var j = 0u;
       if (i == 0u) {
           j = 1u;
       }

       for (; j < max_row_size; j ++) {
           if (new_point_indices[i + j] == 0u) {
               break;
           }
       }
       row_ptr[k] = row_ptr[k - 1u] + j;
       k ++;
    }
}
