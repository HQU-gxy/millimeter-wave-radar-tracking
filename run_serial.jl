import LibSerialPort as Serial
using Logging: @info, @debug, @warn, @error
using Base: @kwdef

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
export t, unmarshal, show
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

const ENDING_MAGIC = UInt8[0x55, 0xcc]
function run_serial()
    Serial.open("/dev/cu.usbserial-0001", 256_000) do port
        Serial.flush(port)
        Serial.set_read_timeout(port, 1)
        Serial.clear_write_timeout(port)
        while true
            data = readuntil(port, ENDING_MAGIC)
            @debug "$(length(data)) $(bytes2hex(data))"
            targets = Targets.unmarshal(data)
            @info "$(targets)"
            Serial.clear_read_timeout(port)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # test_unmarshal()
    run_serial()
end
