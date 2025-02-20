import math
SCRIPT_PATH = "/home/saturn/capn/capn100h/julia_dev/PhotonPropagation/scripts/photon_tables/photon_table_for_sr.jl"
JULIA_PATH =  "/home/hpc/capn/capn100h/.juliaup/bin/julia"
JULIA_DEPOT = "/home/saturn/capn/capn100h/julia_depot"

def calc_niterations(wildcards):
    energy = float(wildcards.energy)
    return math.floor(3600 / ((((math.sqrt(energy) / 612.7095903795534)** 27.847460407675435) + (energy * 0.00012060236341197549)) + 9.095369098636635))


rule run_photon_simulation:
    output:
        r"results/{mode}/output_dist{dist}_energy{energy}_ix{ix}.arrow"
    resources:
        slurm_extra="'--gres=gpu:a40:1'",
        clusters="alex",
        partition="a40",
        time="02:00:00"
    params:
        n_iterations=calc_niterations
    shell:
        "JULIA_DEPOT_PATH={JULIA_DEPOT} julia {SCRIPT_PATH} --outfile {output} --nsims {params.n_iterations} --energy {wildcards.energy} --dist {wildcards.dist} --mode {wildcards.mode}"