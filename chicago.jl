using JuMP
using Cbc
using CSV
using PyPlot

function read_csv(filename)
    raw = CSV.read(filename)
    (rows, cols) = size(raw)
    raw = raw[1:rows, 2:cols]
    data = convert(Matrix{Float64}, raw)
    return data
end

budget = 1200
num_people = 4
min_enjoyment = 150

survey_data = read_csv("survey_data.CSV")
hours_open = read_csv("hours_open.csv")
time_limits = read_csv("time_limits.csv")
costs = read_csv("costs.csv")
activities = convert(Matrix{String}, CSV.read("activities.csv", header=0))
activities = rstrip.(activities)
times = convert(Matrix{String}, CSV.read("times.csv", delim=",", header=0))
times = rstrip.(times)

manditory_hotel_time = [16:25; 40:49]
hotel_indices = 21:23

manditory_food_time = [5; 6; 11; 12; 29; 30; 35; 36; 53; 54; 59; 60]
food_indices = [1; 9; 10; 12:20]

(num_hours, num_activities) = size(hours_open)

function chicago(lam)
    m = Model(with_optimizer(Cbc.Optimizer, logLevel=0))
    
    @variable(m, x[1:num_hours,1:num_activities], Bin)
    @variable(m, lambda[1:num_hours,1:num_activities] >= 0)
    @variable(m, z[1:num_activities], Bin)
    
    # The group is either doing an activity at a certain time or they're not
    @constraint(m, [i=1:num_hours, j=1:num_activities], x[i,j] <= 1)
    
    # z[j] is 1 if the group did activity j during their trip
    @constraint(m, [j=1:num_activities], sum(x[i,j] for i in 1:num_hours) <= num_hours*z[j])
    
    # The group can only do one activity at a time
    @constraint(m, [i=1:num_hours], sum(x[i,j] for j in 1:num_activities) == 1)
    
    # The group cannot do an activity for an unlimited amount of time
    @constraint(m, [j=1:num_activities], sum(x[i,j] for i in 1:num_hours) <= time_limits[j,2])
    
    # If the group does an activity, the must do it for a minimum amount of time
    @constraint(m, [j=1:num_activities], sum(x[i,j] for i in 1:num_hours) >= time_limits[j,1] * z[j])
    
    # The total cost of the trip is less than the group's budget
    @constraint(m,  sum(z[j]*num_people*costs[j] for j in 1:num_activities) <= budget)
    
    # Each person in the group must get some baseline of enjoyment
    @constraint(m, [p=1:num_people], sum(sum(survey_data[p,j]*x[i,j] for i in 1:num_hours) for j in 1:num_activities) >= min_enjoyment)
    
    # The group can only do an activity if it is open
    @constraint(m, [j=1:num_activities], sum(hours_open[i,j]*x[i,j] for i in 1:num_hours) == 0)
    
    # The group must stay at the hotel for sometime each night
    @constraint(m, sum(sum(x[manditory_hotel_time,hotel_indices])) == length(manditory_hotel_time))
    
    # The group can only stay at one hotel during their trip
    @constraint(m, sum(z[j] for j in hotel_indices) == 1)
    
    # The group must eat meals
    @constraint(m, sum(sum(x[manditory_food_time,food_indices])) == length(manditory_food_time))
    
    # The group can only visit an activity (other than the hotel) once
    # lambda 1 one when the group starts or stops an activity
    for j in 1:num_activities
        if j in hotel_indices
            continue
        end
    
        @constraint(m, lambda[1, j] == x[1, j])
    
        for i in 2:num_hours
            @constraint(m, lambda[i,j] >= x[i,j] - x[i-1,j])
            @constraint(m, lambda[i,j] >= -x[i,j] + x[i-1,j])
        end
    end
    @constraint(m, [j=1:num_activities], sum(lambda[i,j] for i in 1:num_hours) <= 2)
    
    # Our objective is to maximize enjoyment!
    @objective(m, Max,
        sum(sum(sum(survey_data[p,j]*x[i,j] for p in 1:num_people) for i in 1:num_hours) for j in 1:num_activities)
            - lam * sum(z[j]*num_people*costs[j] for j in 1:num_activities))
    optimize!(m)

    enjoyment = sum(sum(sum(survey_data[p,j]*value(x[i,j]) for p in 1:num_people) for i in 1:num_hours) for j in 1:num_activities)
    cost = sum(value(z[j])*num_people*costs[j] for j in 1:num_activities)

    println("done one")
    return (enjoyment, cost)
end

lambda = collect(range(0, step=0.2, 3))
enj = []
c = []

for l in lambda
    (enjoyment, cost) = chicago(l)
    push!(enj, enjoyment)
    push!(c, cost)
end

figure()
xlabel("lambda")

plot(lambda, enj)
plot(lambda, c)
