module Medium
using Unitful
using Base: @kwdef
using PhysicalConstants.CODATA2018
using Parquet
using DataFrames
using StructTypes
using PhysicsTools
using AbstractMediumProperties

const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

include("water_properties.jl")
include("ice_properties.jl")
include("simple_medium.jl")


end # Module
