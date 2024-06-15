import { to_words_le } from "../../implementation/cuzk/utils";
import {
  calc_m,
  mp_adder,
  mp_subtracter,
  mp_shifter_left,
  mp_msb_multiply,
  mp_lsb_multiply,
  mp_shifter_right,
  barrett_domb_mul,
  machine_multiply,
  mp_full_multiply,
  machine_two_digit_add,
  mp_lsb_extra_diagonal,
} from "../barrett_domb";
import { genRandomFieldElement } from "../utils";

const p = BigInt(
  "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
);

describe("Barrett-Domb ", () => {
  it("calc_m", () => {
    const word_size = 13;
    const m = calc_m(p, word_size);
    expect(m).toEqual(
      BigInt(
        "3175526949565544461387917976211949610519440472185200499782210609897170941048074",
      ),
    );
  });

  it("mp_lsb_extra_diagonal", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "0x2222222222222222222222222222222222222222222222222222222222222222",
    );
    const b_words = to_words_le(b, num_words, word_size);
    const result = mp_lsb_extra_diagonal(
      a_words,
      b_words,
      num_words,
      word_size,
    );
    expect(result).toEqual(8158);
  });

  it("mp_msb_multiply", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "0x2222222222222222222222222222222222222222222222222222222222222222",
    );
    const b_words = to_words_le(b, num_words, word_size);

    const result = mp_msb_multiply(a_words, b_words, num_words, word_size);
    expect(result.toString()).toEqual(
      [
        18063, 48642, 13689, 44273, 9320, 39904, 4951, 35535, 582, 31166, 61749,
        26796, 57380, 22427, 53011, 1162, 0,
      ].toString(),
    );
  });

  it("mp_msb_multiply 2", () => {
    const num_words = 20;
    const word_size = 13;
    const a = BigInt(
      "12606796758224846727326035948803889824738128609730484894316100840265045196027",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "14276552610056165753848820256553331055663673083569154091093918058913504134283",
    );
    const b_words = to_words_le(b, num_words, word_size);

    const result = mp_msb_multiply(a_words, b_words, num_words, word_size);
    expect(result.toString()).toEqual(
      [
        2165, 4700, 172, 2275, 6372, 2494, 3559, 3643, 4358, 3285, 3136, 6280,
        1860, 5007, 6542, 954, 409, 7593, 3518, 0, 0,
      ].toString(),
    );
  });

  it("mp_lsb_multiply", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "0x2222222222222222222222222222222222222222222222222222222222222222",
    );
    const b_words = to_words_le(b, num_words, word_size);

    const result = mp_lsb_multiply(a_words, b_words, num_words, word_size);
    expect(result.toString()).toEqual(
      [
        3782, 38739, 8155, 43108, 12524, 47477, 16893, 51846, 21262, 56215,
        25631, 60584, 30000, 64953, 34369, 20682, 9905, 0,
      ].toString(),
    );
  });

  it("mp_lsb_multiply 2", () => {
    const num_words = 20;
    const word_size = 13;
    const a = BigInt(
      "12606796758224846727326035948803889824738128609730484894316100840265045196027",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "14276552610056165753848820256553331055663673083569154091093918058913504134283",
    );
    const b_words = to_words_le(b, num_words, word_size);

    const result = mp_lsb_multiply(a_words, b_words, num_words, word_size);
    expect(result.toString()).toEqual(
      [
        2121, 7108, 8052, 630, 3743, 7907, 2443, 7965, 3945, 757, 6249, 4778,
        5546, 4338, 4253, 7207, 1664, 1300, 4815, 6771, 1483, 0,
      ].toString(),
    );
  });

  it("mp_adder", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "0x2222222222222222222222222222222222222222222222222222222222222222",
    );
    const b_words = to_words_le(b, num_words, word_size);

    const result = mp_adder(a_words, b_words, num_words, word_size);
    const expected = to_words_le(a + b, num_words + 1, word_size);
    expect(result.toString()).toEqual(expected.toString());
  });

  it("mp_subtracter", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const b = BigInt(
      "0x1111111111111111111111111111111111111111111111111111111111111111",
    );
    const b_words = to_words_le(b, num_words, word_size);

    const result = mp_subtracter(a_words, b_words, num_words, word_size);
    const expected = to_words_le(a - b, num_words, word_size);
    expect(result.toString()).toEqual(expected.toString());
  });

  it("mp_shifter_left", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const shift = 3;
    const expected = to_words_le(a << BigInt(shift), num_words, word_size);
    const result = mp_shifter_left(a_words, shift, num_words, word_size);
    expect(result.toString()).toEqual(expected.toString());
  });

  it("mp_shifter_left 2", () => {
    const num_words = 20;
    const word_size = 13;
    const a = BigInt(
      "0x2211111111111111111111111111111111111111111111111111111111111133",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const shift = 14;
    const expected = to_words_le(a << BigInt(shift), num_words, word_size);
    const result = mp_shifter_left(a_words, shift, num_words, word_size);
    expect(result.toString()).toEqual(expected.toString());
  });

  it("mp_shifter_right", () => {
    const num_words = 16;
    const word_size = 16;
    const a = BigInt(
      "0x1111111111111111111111111111111111111111111111111111111111111111",
    );
    const a_words = to_words_le(a, num_words, word_size);
    const shift = 3;
    const expected = to_words_le(a >> BigInt(shift), num_words, word_size);
    const result = mp_shifter_right(a_words, shift, num_words, word_size);
    expect(result.toString()).toEqual(expected.toString());
  });

  it("mp_full_multiply with random inputs", () => {
    const num_runs = 100;
    const n = p.toString(2).length;

    for (let word_size = 13; word_size < 17; word_size++) {
      for (let i = 0; i < num_runs; i++) {
        const num_words = Math.ceil(n / word_size);
        const a = genRandomFieldElement(p);
        const b = genRandomFieldElement(p);
        const a_words = to_words_le(a, num_words, word_size);
        const b_words = to_words_le(b, num_words, word_size);
        const result = mp_full_multiply(a_words, b_words, num_words, word_size);
        const expected = to_words_le(a * b, num_words * 2, word_size);
        expect(result.toString()).toEqual(expected.toString());
      }
    }
  });

  it("machine_multiply with random inputs", () => {
    const num_runs = 100;
    for (let word_size = 13; word_size < 17; word_size++) {
      for (let i = 0; i < num_runs; i++) {
        const a = Number(genRandomFieldElement(BigInt(2 ** word_size)));
        const b = Number(genRandomFieldElement(BigInt(2 ** word_size)));
        const res = machine_multiply(a, b, 16);
        const expected = to_words_le(BigInt(a * b), 2, 16);
        expect(res.toString()).toEqual(expected.toString());
      }
    }
  });

  it("machine_two_digit_add with random inputs", () => {
    const num_runs = 100;
    for (let word_size = 13; word_size < 17; word_size++) {
      for (let i = 0; i < num_runs; i++) {
        const a = genRandomFieldElement(BigInt(2) ** BigInt(2 * word_size));
        const a_words = Array.from(to_words_le(a, 2, word_size));
        const b = genRandomFieldElement(BigInt(2) ** BigInt(2 * word_size));
        const b_words = Array.from(to_words_le(b, 2, word_size));

        const expected = to_words_le(a + b, 3, word_size);
        const res = machine_two_digit_add(a_words, b_words, word_size);
        expect(res.toString()).toEqual(expected.toString());
      }
    }
  });

  it("barrett_domb_mul", () => {
    const n = p.toString(2).length;
    const word_size = 13;
    const num_words = Math.ceil(n / word_size);

    const p_words = to_words_le(p, num_words, word_size);

    const a = BigInt(
      "8341461749428370124248824931781546531375899335154063827935293455917439005792",
    );
    const b = BigInt(
      "3444221749422370424228824238781524531374499335144063827945233455917409005733",
    );

    const a_words = to_words_le(a, num_words, word_size);

    const b_words = to_words_le(b, num_words, word_size);

    const m = calc_m(p, word_size);
    const m_words = to_words_le(m, num_words, word_size);

    const result = barrett_domb_mul(
      a_words,
      b_words,
      p_words,
      p.toString(2).length,
      m_words,
      num_words,
      word_size,
    );

    const expected = (a * b) % p;
    const expected_words = to_words_le(expected, num_words, word_size);

    expect(result.toString()).toEqual(expected_words.toString());
  });

  it("barrett_domb_mul with random inputs", () => {
    const num_runs = 1000;
    const word_size = 13;

    const p = BigInt(
      "0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001",
    );
    const n = p.toString(2).length;

    const m = calc_m(p, word_size);

    for (let i = 0; i < num_runs; i++) {
      const num_words = Math.ceil(n / word_size);

      const p_words = to_words_le(p, num_words, word_size);

      const a = genRandomFieldElement(p);
      const a_words = to_words_le(a, num_words, word_size);

      const b = genRandomFieldElement(p);
      const b_words = to_words_le(b, num_words, word_size);

      const m_words = to_words_le(m, num_words, word_size);

      const result = barrett_domb_mul(
        a_words,
        b_words,
        p_words,
        p.toString(2).length,
        m_words,
        num_words,
        word_size,
      );
      const expected = (a * b) % p;
      const expected_words = to_words_le(expected, num_words, word_size);

      expect(result.toString()).toEqual(expected_words.toString());
    }
  });
});
