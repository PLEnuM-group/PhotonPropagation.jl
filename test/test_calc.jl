using Test

# Test closest_approach_distance function
@testset "closest_approach_distance" begin
    # Test case 1
    p0 = [0, 0, 0]
    dir = [1, 0, 0]
    pos = [0, 1, 0]
    expected_distance = 1.0
    @test closest_approach_distance(p0, dir, pos) ≈ expected_distance

    # Test case 2
    p0 = [0, 0, 0]
    dir = [0, 1, 0]
    pos = [1, 1, 0]
    expected_distance = 1.0
    @test closest_approach_distance(p0, dir, pos) ≈ expected_distance

    # Test case 3
    p0 = [0, 0, 0]
    dir = [1, 1, 1]
    pos = [2, 2, 2]
    expected_distance = 0.
    @test closest_approach_distance(p0, dir, pos) ≈ expected_distance
end