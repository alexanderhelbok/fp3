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
    f = 100 * u"mm"
end

# get time to frequency conversion factor
begin
df = CSV.read(joinpath(@__DIR__, "data/T0000.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
df = df[2000:46000, :]
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

plot(df.f, df.CH2)
# scatter(df.f[maxima], df.CH2[maxima], s=100)
plt.show()
end

lorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4]

multilorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10]

begin
    ax = plt.subplots(figsize=(7, 3.5))[1]
    start, stop = 1, 44000
    # mid
    popt, ci = bootstrap(multilorentzian, df.f[start:stop], df.CH2[start:stop], p0=[1., idx_to_f(maxima[1]), 0.006, .4, idx_to_f(maxima[2]), 0.006, 1., idx_to_f(maxima[3]), 0.006, -0.02], redraw=false, unc=true)

    println(popt)

    scatter(df.f[start:stop], df.CH2[start:stop], s=10, label=L"\mathrm{data}")

    ylims = ax.get_ylim()
    x = range(ci.x[1], ci.x[end], length=10000)
    plot(x, multilorentzian(x, nom.(popt)), label=L"\mathrm{multilorentzian\ fit}", color="C1", zorder=5)

    ax.arrow(0.55, 0.5,  0.2, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    ax.arrow(0.55, 0.5, -0.25, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    ax.text(0.45, 0.53, L"\nu_{\mathrm{FSR}} \stackrel{!}{=} 1\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    # ax.arrow(0.4, 0.2,  0.1, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.arrow(0.4, 0.2, -0.1, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.text(0.34, 0.23, L"0.51\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    # ax.arrow(0.65, 0.2,  0.1, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.arrow(0.65, 0.2, -0.11, 0., length_includes_head=true, overhang=.1, head_width=0.03, head_length=0.03, transform=ax.transAxes, fc="white")
    # ax.text(0.59, 0.23, L"0.49\ \mathrm{GHz}", fontsize=14, transform=ax.transAxes)

    ax.set_xlabel(L"\Delta\nu\ (\mathrm{GHz})")
    ax.set_ylabel(L"V\ \mathrm{(arb.u.)}")

    ax.set_xlim(df.f[start], df.f[stop])
    ax.set_ylim(ylims)

    plt.legend()
    plt.tight_layout()
    # savefig(string(@__DIR__, "/bilder/multilorentzian.pdf"), bbox_inches="tight")
    plt.show()
end

# calculate finesse of mirrors
F = FSR / popt[3]
R = (2*F^2 + pi^2 - pi*√(4*F^2 + pi^2))/(2*F^2)

t_to_f([0, 1e-3])

fivelorentzian(x, p) = @. p[1] / ( 1 + ( ( x - p[2] )/ p[3] )^2 ) + p[4] / ( 1 + ( ( x - p[5] )/ p[6] )^2 ) + p[7] / ( 1 + ( ( x - p[8] )/ p[9] )^2 ) + p[10] / ( 1 + ( ( x - p[11] )/ p[12] )^2 ) + p[13] / ( 1 + ( ( x - p[14] )/ p[15] )^2 ) + p[16]
# load data
begin
peaks = []
for i in 1:7
    ax = plt.subplots(figsize=(7, 3.5))[1]
    df = CSV.read(joinpath(@__DIR__, "data/T000$i.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
    # convert data to frequency
    df.f = t_to_f(df.t)
    # smooth data using savgol filter
    df.CH2 = savitzky_golay(df.CH2, 51, 3).y
    # rescale data
    df.CH2 .-= minimum(df.CH2)
    df.CH2 ./= maximum(df.CH2)
    
    if i in [1, 2, 6]
        # find 3 maxima in data
        maxima, height = ss.find_peaks(df.CH2, height=0.09, distance=10000)
        # convert to julia array
        maxima = pyconvert(Array, maxima) .+ 1

        # fit multilorentzian to data
        popt, ci = bootstrap(fivelorentzian, df.f, df.CH2, p0=[.1, df.f[maxima[1]], 0.006, .65, df.f[maxima[2]], 0.006, .8, df.f[maxima[3]], 0.005, .65, df.f[maxima[4]], 0.006, .1, df.f[maxima[5]], 0.006, 0.000], redraw=false, unc=true)
        # add peak heights to peaks array
        plot(df.f, fivelorentzian(df.f, nom.(popt)), c="C2")
    elseif i in [3, 4, 5]
        # find 3 maxima in data
        maxima, height = ss.find_peaks(df.CH2, height=0.07, distance=10000)
        # convert to julia array
        maxima = pyconvert(Array, maxima) .+ 1

        # fit multilorentzian to data
        popt, ci = bootstrap(multilorentzian, df.f, df.CH2, p0=[.65, df.f[maxima[1]], 0.006, .8, df.f[maxima[2]], 0.005, .65, df.f[maxima[3]], 0.006, 0.000], redraw=false, unc=true)
        plot(df.f, multilorentzian(df.f, nom.(popt)), c="C2")
    elseif i in [7]
        popt, ci = bootstrap(lorentzian, df.f, df.CH2, p0=[.65, df.f[argmax(df.CH2)], 0.006, 0.000], redraw=false, unc=true)
        plot(df.f, lorentzian(df.f, nom.(popt)), c="C2")
    end
    push!(peaks, popt[1:3:end-1])
    # plot data
    scatter(df.f, df.CH2, s=5)
    scatter(df.f[maxima], df.CH2[maxima], s=100, c="C1")
    plt.show()
end
end

peaks
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

peaks
newpeaks

besselx = 0:0.01:2

# evaluate Bessels function
begin
# plot sidebands
ax = plt.subplots(3, 1, figsize=(7, 10.5), sharex=true)[1]
for i in 1:3
    ax[i-1].errorbar(x[8-length(newpeaks[i]):end], nom.(newpeaks[i]), yerr=err.(newpeaks[i]), fmt="o")
    bessel = besselj.(2i-2, besselx)
    ax[i-1].plot(2.6*besselx, bessel, label=L"J_{i-1}(x)")
    
    ax[i-1].set_ylabel(latexstring("J_$(2*i-2)(x)"))
end

plt.tight_layout()
# plt.savefig(string(@__DIR__, "/bilder/sidebands.pdf"), bbox_inches="tight")
plt.show()
end


# investigate effect of noise by overplotting data
begin
df = CSV.read(joinpath(@__DIR__, "data/T0012.CSV"), DataFrame, header=["t", "CH1", "peak1", "CH2", "peak2"], skipto=17)
df.f = t_to_f(df.t)

# smooth data using savgol filter
# df.CH2 = savitzky_golay(df.CH2, 51, 3).y

# plot data
ax = plt.subplots(figsize=(7, 3.5))[1]
scatter(df.f, df.CH2, s=5)

plt.show()
end