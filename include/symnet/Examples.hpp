// Examples.hpp - built-in SymNet DSL demonstration programs (Phase 1 and Phase 2).
//
// These are the same programs as the demonstrations in the original symnet.py. They are used
// by the CLI (`symnet_cli`) when no file is given and by the unit tests.
#pragma once

namespace symnet {
namespace examples {

// =====================================================================================

inline constexpr const char* KLEIN_XOR_SOURCE = R"(
# Phase 1: hand-written lift z = |x - y|
group Klein4 {
    lift z = abs(x - y)
    separate XorClassifier { weight = [0, 0, 1], bias = -0.5 }
    classify XorClassifier on (0, 0) expect 0
    classify XorClassifier on (1, 0) expect 1
    classify XorClassifier on (0, 1) expect 1
    classify XorClassifier on (1, 1) expect 0
}
)";

inline constexpr const char* D4_SOURCE = R"(
# Phase 1: dihedral group D4, word r*s*r, subgroup H = {e, s}
group D4 {
    gen r = rotation(90)
    gen s = matrix([[-1, 0], [0, 1]])
    subgroup D = span(r, s)
    subgroup H = span(s)
    separate InH { weight = [0, 0, 0, 1], bias = -0.5 }
    eval r * s * r
    classify InH of r * s * r expect 1
    verify InH = H over D
}
)";

inline constexpr const char* XOR_SPACE_SOURCE = R"(
# 1a. Raw XOR point set.  NO lift and NO weights are written by hand:
#     the synthesizer discovers the lift and the hyperplane.
group XORSpace {
    space Points {
        (0, 0) class 0
        (1, 0) class 1
        (0, 1) class 1
        (1, 1) class 0
    }
    synthesize XorAuto on Points
    classify XorAuto on (1, 0) expect 1       # ordinary DSL queries use the auto-built pipeline
    classify XorAuto on (1, 1) expect 0
}
)";

inline constexpr const char* KLEIN_GROUP_SOURCE = R"(
# 1b. Klein 4-group V = {I, a, b, c=ab} as 2x2 matrices, with subgroup H = {I, c}.
#     On vec(M) = (m00, m01, m10, m11) the partition {I,c} | {a,b} is XOR in disguise.
group Klein4Matrices {
    gen a = matrix([[-1, 0], [0, 1]])
    gen b = matrix([[1, 0], [0, -1]])
    gen c = matrix([[-1, 0], [0, -1]])
    subgroup V = span(a, b)
    subgroup H = span(c)

    synthesize InH of H over V
    synthesize InCosetA of H over V coset a      # left coset aH = {a, b}

    classify InH of a * b expect 1
    classify InH of a expect 0
    classify InCosetA of b expect 1
    classify InCosetA of c expect 0
}
)";

inline constexpr const char* D4_CENTER_SOURCE = R"(
# 1c. Centre Z(D4) = {e, r^2} sits between the other elements in R^4 (the origin lies
#     in the convex hulls of both classes), so no hyperplane works until a lift is added.
group D4Center {
    gen r = rotation(90)
    gen s = matrix([[-1, 0], [0, 1]])
    gen c = rotation(180)
    subgroup D = span(r, s)
    subgroup Z = span(c)
    synthesize InCenter of Z over D
    classify InCenter of r * r expect 1
    classify InCenter of s expect 0
    classify InCenter of r * s expect 0
}
)";

inline constexpr const char* SO2_SOURCE = R"(
# 2. Continuous rotation group SO(2) = U(1), generators R(theta) = e^(i theta).
group ContinuousSO2 {
    gen R(theta) = phasor(theta)                 # radians

    sector Wedge   = [pi/4, 3*pi/4]              # width pi/2 : two half-planes AND-ed
    sector Reflex  = [pi/2, 2*pi]                # width 3pi/2: two half-planes OR-ed

    eval R(pi/4) * R(pi/4)                       # composition of two phasors
    check R(pi/4) * R(pi/4) == R(pi/2)           # must equal [[0,-1],[1,0]] = i
    eval R(pi/4) * R(pi/4) on (1, 0)             # action on the vector (1, 0)

    classify Wedge  of R(pi/4) * R(pi/4) on (1, 0) expect 1    # lands at 90 deg: inside
    classify Wedge  of R(pi/8) on (1, 0) expect 0              # 22.5 deg: outside
    classify Reflex of R(pi/4) on (1, 0) expect 0              # 45 deg: outside

    sweep Wedge  over R from 0 to 2*pi steps 8   # 0, 45, ..., 315 deg (45 and 135 are edges)
    sweep Reflex over R from 0 to 2*pi steps 8
}
)";

struct Example {
    const char* name;
    const char* source;
};

// Phase 1 followed by Phase 2, in the order of the original script.
inline constexpr Example kExamples[] = {
    {"Klein4 hand-written lift", KLEIN_XOR_SOURCE},
    {"D4 word r*s*r and subgroup H", D4_SOURCE},
    {"XOR point set (auto lift)", XOR_SPACE_SOURCE},
    {"Klein 4-group matrices (auto lift)", KLEIN_GROUP_SOURCE},
    {"D4 centre Z(D4) (auto lift)", D4_CENTER_SOURCE},
    {"Continuous SO(2)", SO2_SOURCE},
};

}  // namespace examples
}  // namespace symnet
