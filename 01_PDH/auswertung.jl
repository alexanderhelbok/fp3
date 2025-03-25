using CSV, Roots, LaTeXStrings, PythonCall, Bessels, SavitzkyGolay
using PhysicalConstants.CODATA2018: c_0

include(string(@__DIR__, "/../Source.jl"))
@py import matplotlib as mpl
@py import scipy.signal as ss
@py import matplotlib.pyplot as plt

mpl.use("pgf")
mpl.use("TkAgg")

begin
    λ = 633 * u"nm"
    r = 150 * u"mm"
    L = 150 * u"mm"
end

# get time to frequency conversion factor
begin
df = CSV.read(joinpath(@__DIR__, "data/T0000.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
df = df[5000:43000, :]
# shift data by minimum
df.CH2 .-= mean(df.CH2[1:500])
# normalize data
df.CH2 /= maximum(df.CH2)

FSR = c_0/(2*L) / u"GHz"|> NoUnits

# find 2 maxima in the data
maxima, height = ss.find_peaks(df.CH2, height=0.04, distance=2000)
# convert to julia array
maxima = pyconvert(Array, maxima) .+ 1

dt = df.t[maxima[3]] - df.t[maxima[1]]

t_to_f(t) = @. FSR / dt * (t - t[1])
df.f = t_to_f(df.t)

didx = maxima[3] - maxima[1]

idx_to_f(idx) = @. FSR / didx * (idx - 1)
f_to_idx(f) = @. didx / FSR * f + 1 |> round |> Int

# popt, perr, ci = bootstrap(pseudo_voigt, t_to_f(df.t), df.CH2, p0=[1, 0.4, 0.5, 0.1, 0.1], redraw=true)

# plot(df.f, df.CH2)
# scatter(df.f[maxima], df.CH2[maxima], s=100)
# plt.show()
end

lorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4]
threelorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10]
fivelorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10] / ( 1 + ( ( x - p[11] )/ p[12] )^2 ) + p[13] / ( 1 + ( ( x - p[14] )/ p[15] )^2 ) + p[16]

begin
    ax = plt.subplots(figsize=(7, 3.5))[1]
    start, stop = 1, 38000
    # mid
    popt, ci = bootstrap(threelorentzian, df.f[start:stop], df.CH2[start:stop], p0=[1., idx_to_f(maxima[1]), 0.006, .4, idx_to_f(maxima[2]), 0.006, 1., idx_to_f(maxima[3]), 0.006, -0.02], redraw=false, unc=true)

    println(popt)

    scatter(df.f[start:stop], df.CH2[start:stop], s=10, label=L"\mathrm{data}")

    ylims = ax.get_ylim()
    x = range(ci.x[1], ci.x[end], length=10000)
    plot(x, threelorentzian(x, nom.(popt)), label=L"\mathrm{multilorentzian\ fit}", color="C1", zorder=5)

    ax.arrow(0.55, 0.53,  0.275, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    ax.arrow(0.55, 0.53, -0.345, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    ax.text(0.45, 0.56, L"\nu_{\mathrm{FSR}} \stackrel{!}{=} 1\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)
    
    ax.text(0.4, 0.36, L"F = 81.11(7)", fontsize=14, transform=ax.transAxes)
    ax.text(0.4, 0.26, L"R = 96.201(3)\ \%", fontsize=14, transform=ax.transAxes)
    rect = mpl.patches.FancyBboxPatch((0.4, 0.23), 0.225, 0.2, ec="C1", fc="none", transform=ax.transAxes, boxstyle=mpl.patches.BoxStyle("Round", pad=0.02))
    ax.add_patch(rect)

    ax.set_xlabel(L"\Delta\nu\ (\mathrm{GHz})")
    ax.set_ylabel(L"V\ \mathrm{(arb.u.)}")

    ax.set_xlim(df.f[start], df.f[stop])
    ax.set_ylim(ylims)

    plt.legend()
    plt.tight_layout()
    # plt.savefig(string(@__DIR__, "/bilder/FSR.pdf"), bbox_inches="tight")
    plt.show()
end

# calculate finesse of mirrors
FWHM = popt[3] + popt[9]
F = FSR / FWHM
R = (2*F^2 + pi^2 - pi*√(4*F^2 + pi^2))/(2*F^2)

# load data
begin
ax = plt.subplots(4, 1, figsize=(7, 6), constrained_layout=true)[1]
peaks = []
for i in 1:7
    df = CSV.read(joinpath(@__DIR__, "data/T000$i.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
    # convert data to frequency
    df.f = t_to_f(df.t)*1e3
    # smooth data using savgol filter
    df.CH2 = savitzky_golay(df.CH2, 51, 3).y
    # rescale data
    df.CH2 .-= minimum(df.CH2)
    df.CH2 ./= maximum(df.CH2)
    
    if i in [1, 2, 6]
        maxima, height = ss.find_peaks(df.CH2, height=0.09, distance=10000)
        maxima = pyconvert(Array, maxima) .+ 1

        model = fivelorentzian
        popt, ci = bootstrap(model, df.f, df.CH2, p0=[.1, df.f[maxima[1]], 6., .65, df.f[maxima[2]], 6., .8, df.f[maxima[3]], 5., .65, df.f[maxima[4]], 6., .1, df.f[maxima[5]], 6., 0.000], redraw=false, unc=true)
    elseif i in [3, 4, 5]
        maxima, height = ss.find_peaks(df.CH2, height=0.07, distance=10000)
        maxima = pyconvert(Array, maxima) .+ 1

        model = threelorentzian
        popt, ci = bootstrap(model, df.f, df.CH2, p0=[.65, df.f[maxima[1]], 6., .8, df.f[maxima[2]], 0.005, .65, df.f[maxima[3]], 6., 0.000], redraw=false, unc=true)
    elseif i in [7]
        model = lorentzian
        popt, ci = bootstrap(model, df.f, df.CH2, p0=[.65, df.f[argmax(df.CH2)], 6., 0.000], redraw=false, unc=true)
    end
    push!(peaks, popt[1:3:end-1] .- popt[end])
    # plot data
    if i == 1
        ax[0].scatter(df.f, df.CH2, s=5)
        ax[0].plot(df.f, model(df.f, nom.(popt)), c="C1")
    end
    # scatter(df.f[maxima], df.CH2[maxima], s=100, c="C1")
end
ax[0].set_xlim(0, 210)
ax[0].set_xlabel(L"\Delta\nu\ (\mathrm{MHz})")
ax[0].set_ylabel(L"V\ \mathrm{(arb.u.)}")

x = [5, 4, 3, 2, 1, 4.5, 0]
# sort peaks according to x
peaks = [peaks[i] for i in sortperm(x)]
x = sort(x)
# normalize peaks by the sum
peaks = [peak ./ sum(peak) for peak in peaks]

newpeaks = [[], [], []]
# combine opposing peaks due to symmetry
for (i, peak_arr) in enumerate(peaks)
    push!(newpeaks[1], peak_arr[ceil(Int, length(peak_arr)/2)])
    for j in 1:floor(Int, length(peak_arr)/2)
        if i < 5
            push!(newpeaks[j+1], (peak_arr[j] + peak_arr[end-j+1])/2)
        else
            push!(newpeaks[4-j], (peak_arr[j] + peak_arr[end-j+1])/2)
        end
    end
end

besselx = 0:0.01:6

# plot sidebands
popts = []
for i in 1:3
    ax[i].errorbar(x[8-length(newpeaks[i]):end], nom.(newpeaks[i]), yerr=err.(newpeaks[i]), fmt="o", mfc="C0", mec="k", ecolor="k")
    # fit bessel function
    model(x, p) = besselj.(2i-2, x/p[1])
    popt, ci = bootstrap(model, x[8-length(newpeaks[i]):end], nom.(newpeaks[i]), p0=[2.5], redraw=false, unc=true)
    println(popt)
    push!(popts, popt)

    ax[i].plot(besselx, model(besselx, nom.(2.8)), label=L"\mathrm{fit}")
    
    ax[i].set_ylabel(latexstring("J_$(2*i-2)(x)"))

    ax[i].set_xlim(0, 5.25)
end
popt = mean(popts)
println(popt)

# turn off xticks for all but the last plot
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])

# plt.tight_layout()
plt.xlabel(L"\mathrm{Modulation\ Amplitude\ (Vpp)}")
# plt.savefig(string(@__DIR__, "/bilder/sidebands.pdf"), bbox_inches="tight")
plt.show()
end


# investigate effect of noise by overplotting data
Vpplabel = [L"0.4\ \mathrm{Vpp}", L"0.5\ \mathrm{Vpp}", L"0.6\ \mathrm{Vpp}", L"0.7\ \mathrm{Vpp}", L"1.5\ \mathrm{Vpp}"]
begin
ax = plt.subplots(figsize=(7, 3.5))[1]
for (i, j) in enumerate([11,12,10,9,8])
    if j < 10
        j = "0$j"
    end
    df = CSV.read(joinpath(@__DIR__, "data/T00$j.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
    df.f = t_to_f(df.t) .* 1e3

    # identify min and max
    minidx, maxidx = argmin(df.CH2), argmax(df.CH2)
    # sort min max so that min is first
    maxidx, minidx = sort([minidx, maxidx])

    # find first datapoint that reaches 0 coming from both sides
    min0 = findfirst(df.CH2[maxidx:end] .<= 0.0001) + maxidx - 1
    max0 = -findfirst(df.CH2[minidx:-1:maxidx] .>= -0.0001) + minidx + 1
    center0 = (min0 + max0) ÷ 2

    # shift data by center
    df.f .-= df.f[center0]

    # smooth data and calculate mean deviation
    smooth_y = savitzky_golay(df.CH2, 5001, 2).y
    deviation = mean(abs.(df.CH2 .- smooth_y))
    println("Vpp = $(Vpplabel[i]), deviation = $deviation")

    # plot data
    scatter(df.f[1:10:end], df.CH2[1:10:end], s=1)
end
# create dummy plot for legend
for i in 1:5
    scatter([], [], label=Vpplabel[i], c="C$(5-i)")
end

plt.xlim(-90, 90)
plt.xlabel(L"\Delta\nu\ (\mathrm{MHz})")
plt.ylabel(L"A\ \mathrm{(arb.u.)}")
leg = plt.legend(ncols=5, columnspacing=0.25, handletextpad=-0.25, loc="upper center", bbox_to_anchor=(0.5, 1.3))
for legmarker in leg.legend_handles
    legmarker.set_sizes([10])
end
plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/phasenoise.pdf"), bbox_inches="tight")
plt.show()
end

# construct logspace by placing five datapoints in each decade
start = log10(10)
stop = log10(10000)
num_points = 25  # Adjust this value to change the number of points

colors = ["C0", "C1", "C4"]
logspace = 10 .^ range(start, stop, length=num_points)
begin
ax = plt.subplots(1, 2, figsize=(7, 3.5))[1]
df = CSV.read(joinpath(@__DIR__, "data/T0012.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
df.f = t_to_f(df.t) .* 1e3

# identify min and max
minidx, maxidx = argmin(df.CH2), argmax(df.CH2)
# sort min max so that min is first
maxidx, minidx = sort([minidx, maxidx])

# find first datapoint that reaches 0 coming from both sides
min0 = findfirst(df.CH2[maxidx:end] .<= 0.0001) + maxidx - 1
max0 = -findfirst(df.CH2[minidx:-1:maxidx] .>= -0.0001) + minidx + 1
center0 = (min0 + max0) ÷ 2

# shift data by center
df.f .-= df.f[center0]

# plot data
i = 0
for (k, window) in enumerate(logspace)
    # round logrange to next odd number
    window = round(Int, window)
    if window % 2 == 0
        window += 1
    end
    # smooth data
    smooth_y = savitzky_golay(df.CH2, window, 2).y
    # calculate mean deviation from smoothed data
    deviation = mean(abs.(df.CH2 .- smooth_y))
    
    ax[1].scatter(window, deviation, s=35, ec="k", marker="o", fc="none")
    if k in [4, 19, 24]
        i += 1
        ax[0].plot(df.f, smooth_y, label="window = $window", c=colors[i])
        ax[1].scatter(window, deviation, label=latexstring("\\mathrm{window} = $window"), s=35, ec="k", marker="o", c=colors[i])
    end
ax[0].scatter(df.f, df.CH2, label=L"\mathrm{Data}", s=1, c="k")
end
ax[0].set_xlim(-75, 0)
ax[0].set_ylim(bottom=-0.012)
ax[1].set_xscale("log")

ax[0].set_xlabel(L"\Delta\nu\ (\mathrm{MHz})")
ax[0].set_ylabel(L"A\ \mathrm{(arb.u.)}")
ax[1].set_xlabel(L"\mathrm{Window\ Size}")
ax[1].set_ylabel(L"\mathrm{Mean\ Deviation}")

# ax1 scientific notation for yaxis ticks
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax[1].legend()
plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/smoothing.pdf"), bbox_inches="tight")
plt.show()
end


