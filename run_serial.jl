import LibSerialPort as Serial
using Logging: @info, @debug, @warn, @error
using Base: @kwdef
using DataStructures: Deque

# https://github.com/JuliaLang/julia/issues/1255
# https://en.wikipedia.org/wiki/Multiple_dispatch
# https://thautwarm.github.io/MLStyle.jl/latest/syntax/pattern.html

module Target
export coord_si, speed_si, t, unmarshal, show
@kwdef struct t
    """
    in millimeters, Int16 (MSB is the sign bit) in little endian
    """
    coord::Tuple{Int16,Int16}

    """
    only magnitude, in cm/s, Int16 (MSB is the sign bit)
    """
    speed::Int16

    resolution::UInt16

    function t(coord, speed, resolution)
        new(coord, speed, resolution)
    end
end

@inline function coord_si(target::t)::Tuple{Float32,Float32}
    x, y = target.coord
    return (x / 1000, y / 1000)
end

@inline function speed_si(target::t)::Float32
    return target.speed / 100
end

@inline function msb_bit_int16(num::Integer)
    """
    Some genius decided to use the most significant bit as the sign bit

    Parameters:
        num (Integer): A 16-bit number (unsigned)
    Returns:
        Int16: A 16-bit number (use most significant bit as sign bit)
    """
    @assert 0 <= num < 2^16 "Number out of range for 16-bit unsigned integer"
    sign = num & 0x8000 >= 1
    n = num & 0x7fff
    x = unsafe_trunc(Int16, n)
    return sign ? x : -x
end

function Base.show(io::IO, target::t)
    x, y = target.coord
    speed = target.speed
    resolution = target.resolution
    println(io, "Target(coord=($x, $y), speed=$speed, resolution=$resolution)")
end

function unmarshal(data::AbstractArray{UInt8})::Union{t,Nothing}
    @assert length(data) == 8 "Invalid data length"
    if all(data .== 0)
        return nothing
    end

    x_ = reinterpret(UInt16, data[1:2])[1]
    x = msb_bit_int16(x_)

    y_ = reinterpret(UInt16, data[3:4])[1]
    y = msb_bit_int16(y_)

    speed_ = reinterpret(UInt16, data[5:6])[1]
    speed = msb_bit_int16(speed_)

    resolution = reinterpret(UInt16, data[7:8])[1]

    return t((x, y), speed, resolution)
end
end

module Targets
import ..Target
export t, unmarshal, show, ENDING_MAGIC
@kwdef struct t
    targets::Vector{Target.t} = Target.t[]
end

function Base.show(io::IO, targets::t)
    print(io, "Targets[")
    i = 1
    for target in targets.targets
        target_string = replace(string(target), r"\n" => "")
        print(io, "$target_string")

        if i < length(targets.targets)
            print(io, ", ")
        end
        i += 1
    end
    print(io, "]")
end

const TARGETS_MAGIC = UInt8[0xaa, 0xff, 0x03, 0x00]
const ENDING_MAGIC = UInt8[0x55, 0xcc]
function unmarshal(data::AbstractArray{UInt8})::t
    offset = 1
    if data[1:4] != TARGETS_MAGIC
        throw(ArgumentError("Invalid magic"))
    end
    offset += 4
    targets = Target.t[]
    # we have three targets
    # if the data is all zeros, it means the target is not set
    for _ in 1:3
        target = Target.unmarshal(data[offset:offset+7])
        if !isnothing(target)
            push!(targets, target)
        end
        offset += 8
    end
    return t(targets)
end
end

module TargetFIS
export targetsWindow2Fuzzy
using FuzzyLogic
using Statistics
fis = @mamfis function decider(xAvg, yAvg, speedMean, speedStd)::output
    xAvg := begin
        domain = -1000:1000
        # left
        XL = GaussianMF(-1000.0, 200.0)
        # optimal
        XO = GaussianMF(-220.0, 180.0)
        # right
        XR = GaussianMF(500.0, 230.0)
    end
    yAvg := begin
        domain = -10:2000
        # optimal
        YO = GaussianMF(0.0, 750.0)
        # negative
        YN = GaussianMF(2000.0, 250.0)
    end
    speedMean := begin
        domain = 0:12
        # optimal
        SMO = ZShapeMF(6, 12)
        # high
        SMH = GaussianMF(15.0, 2.0)
    end
    speedStd := begin
        domain = 0:8
        # optimal
        SSO = ZShapeMF(4, 8)
        # high
        SSH = GaussianMF(8.0, 1.2)
    end
    output := begin
        domain = -1:1
        # false
        OF = GaussianMF(-1.0, 0.6)
        # true
        OT = GaussianMF(1.0, 0.6)
    end

    and = ProdAnd
    or = ProbSumOr
    implication = ProdImplication

    xAvg == XO && yAvg == YO && speedMean == SMO && speedStd == SSO --> output == OT
    # Others are default to L
    xAvg == XL --> output == OF
    xAvg == XR --> output == OF
    yAvg == YN --> output == OF
    speedMean == SMH --> output == OF
    speedStd == SSH --> output == OF
    # additional rules
    xAvg == XO && yAvg == YN --> output == OF
    xAvg == XL && speedMean == SMO --> output == OF
    xAvg == XR && speedMean == SMO --> output == OF
    yAvg == YO && speedMean == SMH --> output == OF
    yAvg == YO && speedStd == SSH --> output == OF

    aggregator = ProbSumAggregator
    defuzzifier = CentroidDefuzzifier
end

"""
Find the centroid of a function `mf` in the domain `domain` by approximating the integral.

# Arguments

- `domain::UnitRange{Int}` The domain of the function.
- `mf::(x::Real)::Real` The function to find the centroid.
- `segmentation::Int` The number of segments to divide the domain, used to approximate the integral.
"""
function centroid(
    domain::UnitRange{Int},
    mf::Any,
    segmentation::Int=100,
)::Real
    x = LinRange(first(domain), last(domain), segmentation)
    dx = (last(domain) - first(domain)) / (segmentation - 1)
    y = map(mf, x)
    trapz(dx, y) = (2sum(y) - first(y) - last(y)) * dx / 2
    ∫f = trapz(dx, y)
    ∫xf = trapz(dx, x .* y)
    centroid_x = ∫xf / ∫f
    centroid_x
end

function mapRange(
    x::Real,
    inMin::Real,
    inMax::Real,
    outMin::Real,
    outMax::Real,
)::Real
    (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin
end

import ..Target


const maxVal = centroid(-1:1, GaussianMF(1.0, 0.6))
const minVal = centroid(-1:1, GaussianMF(-1.0, 0.6))
function targetsWindow2Fuzzy(
    target::AbstractArray{Target.t},
)::Real
    xAvg = map(x -> x.coord[1], target) |> mean
    yAvg = map(x -> x.coord[2], target) |> mean
    # note that speed should be absolute value
    speed = map(x -> x.speed |> abs, target)

    speedAvg = mean(speed)
    speedStd = std(speed)

    calc(xAvg, yAvg, speedAvg, speedStd) = fis(xAvg=xAvg, yAvg=yAvg, speedMean=speedAvg, speedStd=speedStd)[:output] |> x -> mapRange(x, minVal, maxVal, -1, 1)
    calc(xAvg, yAvg, speedAvg, speedStd)
end
end

function test_unmarshal()
    data = UInt8[
        0xaa, 0xff, 0x03, 0x00, 0x0e, 0x03, 0xb1, 0x86, 0x10, 0x00, 0x40, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    ]
    targets = Targets.unmarshal(data)
    @assert length(targets.targets) == 1
    @assert targets.targets[1].coord == (-782, 1713)
    @assert targets.targets[1].speed == -16
    @assert targets.targets[1].resolution == 320
    @info targets
end

function bytes2hex(arr::AbstractArray{UInt8})::String
    hex(i, n) = string(i, base=16, pad=n)
    join(["0x$(hex(i, 2)) " for i in arr])
end

# https://stackoverflow.com/questions/64623810/julia-generics-function-type-parameter
const QUEUE_SIZE = 16
function roll(
    window::Deque{T},
    item::T,
    size::Int)::Nothing where {T}
    if length(window) >= size
        popfirst!(window)
    end
    push!(window, item)
    nothing
end

function run_serial()
    # https://en.wikibooks.org/wiki/Introducing_Julia/Controlling_the_flow#Do_block
    window = Deque{Target.t}(QUEUE_SIZE)
    Serial.open("/dev/cu.usbserial-0001", 256_000) do port
        Serial.flush(port)
        Serial.set_read_timeout(port, 1)
        Serial.clear_write_timeout(port)

        function calc(data::Targets.t)::Union{Nothing,Real}
            if isempty(data.targets)
                return nothing
            end
            fstTarget = first(data.targets)
            roll(window, fstTarget, QUEUE_SIZE)
            if length(window) != QUEUE_SIZE
                return nothing
            end
            return TargetFIS.targetsWindow2Fuzzy(collect(window))
        end

        while true
            data = readuntil(port, Targets.ENDING_MAGIC)
            Serial.clear_read_timeout(port)
            @debug "$(length(data)) $(bytes2hex(data))"
            targets = Targets.unmarshal(data)
            @info "$(targets)"

            result = calc(targets)
            if !isnothing(result)
                @info "FIS result: $result"
            end
        end
    end
end

function main()
    run_serial()
end

# https://stackoverflow.com/questions/74022765/is-there-any-similar-method-of-if-name-main-in-julia
# https://discourse.julialang.org/t/julia-python-equivalent-of-main/35433/2
# https://docs.julialang.org/en/v1/manual/faq/#How-do-I-check-if-the-current-file-is-being-run-as-the-main-script?
# https://towardsdatascience.com/1-length-a-considered-harmful-or-how-to-make-julia-code-generic-safe-ac7b39cfc2f0
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
