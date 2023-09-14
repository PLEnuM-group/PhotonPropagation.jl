using CairoMakie
using Polynomials
using PhotonPropagation
using StaticArrays
using Random
using Distributions
using PhysicsTools
using Rotations
using CSV
using DataFrames
using LinearAlgebra




pos = SA_F32[0., 0., 0.]
dom = DOM(pos, 1)

integrate_gauss_quad(x -> dom.acceptance.poly_ang(x), -1., 1.)

p_int = integrate(dom.acceptance.poly_ang)

p_int(1) - p_int(-1)

DOMAcceptance

n= 10000

uni_zen = acos.(rand(Uniform(-1, 1), n))
uni_azi = rand(Uniform(0, 2*π), n)
uni_cart = sph_to_cart.(uni_zen, uni_azi)

acc = check_pmt_hit(uni_cart, uni_cart, uni_cart, ones(n), dom, RotMatrix3(I))

fig, ax, h = hist(cos.(uni_zen)[acc])
xs = -1:0.01:1
lines!(ax, xs, dom.acceptance.poly.(xs) .* n)
fig

lines(xs, dom.acceptance.poly_ang.(xs))

wls = 300:1:800

lines(wls, dom.acceptance.int_wl(wls))
