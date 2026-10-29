"""INT4 权重的打包与快速反量化。对应 2.6 节。

演示两件事：
1. 8 个 4 位整数怎样打包进一个 32 位整数，checkpoint 里的 qweight 就是这样存的；
2. 低比特 kernel 怎样不用整数到浮点的转换指令，用位运算把 4 位整数变成 FP16：
   把 4 位数 u 放进 FP16 0x6400（即 1024.0）的尾数低 4 位，得到的 FP16 恰好是 1024 + u，
   再减去 1024 + 8（对称量化的零点）就得到 u - 8。

    python tools/int4_pack.py
"""
import numpy as np


def pack_int4(q: np.ndarray) -> np.ndarray:
    """q: 取值 0..15 的整数数组，最后一维长度是 8 的倍数。返回 uint32，最后一维缩小为 1/8。
    第 i 个数放在第 4i 到 4i+3 位，与 GPTQ checkpoint 的 qweight 打包顺序相同。"""
    q = np.asarray(q, dtype=np.uint32)
    assert q.shape[-1] % 8 == 0 and q.max() < 16
    q = q.reshape(*q.shape[:-1], -1, 8)
    shifts = np.arange(8, dtype=np.uint32) * 4
    return np.bitwise_or.reduce(q << shifts, axis=-1)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    shifts = np.arange(8, dtype=np.uint32) * 4
    q = (packed[..., None] >> shifts) & 0xF
    return q.reshape(*packed.shape[:-1], -1)


def magic_dequant(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """用位运算把打包的 INT4 变成 FP16 并乘以 scale。scale 的形状要能广播到解包后的结果。"""
    shifts = np.arange(8, dtype=np.uint32) * 4
    nibbles = ((packed[..., None] >> shifts) & 0xF).astype(np.uint16)
    bits = nibbles | np.uint16(0x6400)                # FP16 的 1024.0，尾数低 4 位放入 u
    as_fp16 = bits.view(np.float16)                   # 数值恰好是 1024 + u
    w = (as_fp16 - np.float16(1032.0))                # 减去 1024 和零点 8
    w = w.reshape(*packed.shape[:-1], -1)
    return (w * scale).astype(np.float16)


def main():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 32)).astype(np.float32) * 0.02
    scale = np.abs(W).max(axis=1, keepdims=True) / 7
    q = np.clip(np.round(W / scale), -8, 7).astype(np.int32) + 8          # 存成 0..15
    packed = pack_int4(q)
    print(f"原始权重 {W.shape}，FP16 共 {W.size * 2} 字节；打包后 {packed.shape}，共 {packed.nbytes} 字节")
    print(f"第 0 行的前 8 个量化值：{q[0, :8].tolist()}  打包成 0x{int(packed[0, 0]):08x}")
    assert (unpack_int4(packed) == q).all()

    ref = ((q - 8) * scale).astype(np.float16)
    fast = magic_dequant(packed, scale.astype(np.float16))
    print(f"位运算反量化与直接计算的最大差：{np.abs(fast.astype(np.float32) - ref.astype(np.float32)).max():.2e}")
    u = np.arange(16, dtype=np.uint16)
    print("0x6400 | u 作为 FP16 的值：", (u | np.uint16(0x6400)).view(np.float16).astype(int).tolist())


if __name__ == "__main__":
    main()
