type_aliases = {'int8_t':'i8', 'uint8_t': 'u8', 'int16_t':'i16', 'uint16_t':'u16', 'int32_t': 'i32', 'uint32_t': 'u32', 'int64_t': 'i64', 'uint64_t': 'u64', 'float': 'f32', 'double': 'f64'}
def alias(t, c):
    return f"{type_aliases[t]}x{c}"

size_classes = {'byte': 'b', 'word': 'w', 'dword': 'd', 'qword': 'q'}
counts = [2**i for i in range(1, 7)]

for c in counts:
    for t in size_classes.keys():
        print(f"typedef SIMD_Mask<concepts::LaneSizeEnum::{t}, {c}> mask{c}{size_classes[t]};")
    print()