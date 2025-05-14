using CSV, Roots, LaTeXStrings, PythonCall, SavitzkyGolay
using PhysicalConstants.CODATA2018: c_0

include(string(@__DIR__, "/../Source.jl"))
@py import matplotlib as mpl
@py import scipy.signal as ss
@py import matplotlib.pyplot as plt

mpl.use("pgf")
mpl.use("TkAgg")


function draw_insert_from(fig, ax1, ax2; lines_to_draw=[1, 1, 1, 1], color="black", alpha=0.3)
    xl, yl =ax2.get_xlim(), ax2.get_ylim()
    top_left = (xl[0], yl[1])
    top_right = (xl[1], yl[1])
    bottom_right = (xl[1], yl[0])
    bottom_left = (xl[0], yl[0])
    corners = [top_left, top_right, bottom_right, bottom_left]
    lines = []

    transFigure = fig.transFigure.inverted()
    for i in 1:4
        j = i%4 + 1
        println(i, j)
        x1, y1 = corners[i]
        x2, y2 = corners[j]
        coord1 = transFigure.transform(ax1.transData.transform([x1, y1]))
        coord2 = transFigure.transform(ax1.transData.transform([x2, y2]))
        push!(lines, mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure, color=color, alpha=alpha))
    end

    for (i, corner) in enumerate(corners)
        if lines_to_draw[i] == 1
            x, y = corner
            coord1 = transFigure.transform(ax2.transData.transform([x, y]))
            coord2 = transFigure.transform(ax1.transData.transform([x,y]))
            push!(lines, mpl.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure, color=color, alpha=alpha))
        end
    end

    fig.lines = lines
end

df = CSV.read(joinpath(@__DIR__, "data/birnefft.trc"), DataFrame, header=["Ax", "Ay"], skipto=17)
df2 = CSV.read(joinpath(@__DIR__, "data/birnefft2.trc"), DataFrame, header=["Ax", "Ay"], skipto=17)
df.Ax /= 1000
df2.Ax /= 1000
# plot ax vs Ay
begin
fig, ax = plt.subplots(2, 1, figsize=(7.5, 4.25))

for axis in ax
    axis.plot(df.Ax, df.Ay, label=L"\mathrm{Raw}")
    axis.plot(df2.Ax, df2.Ay, label=L"\mathrm{Filter}")
    axis.set_xlabel(L"f\ (\mathrm{kHz})")
    axis.set_ylabel(L"A\ (\mathrm{dBV})")
    axis.grid()
end

plt.tight_layout()

# ax[1].set_ylim(-35, -5)
ax[1].set_xlim(0, 5)


draw_insert_from(fig, ax[0], ax[1], lines_to_draw=[1, 1, 0, 0])
plt.legend(borderaxespad=0.5)
# plt.savefig(joinpath(@__DIR__, "bilder/birnefft.pdf"), bbox_inches="tight")
plt.show()
end

df = CSV.read(joinpath(@__DIR__, "data/led fft.trc"), DataFrame, header=["Ax", "Ay"], skipto=17)
df.Ax /= 1000
# plot ax vs Ay
begin
fig, ax = plt.subplots(figsize=(7.5, 3.5))
ax.plot(df.Ax, df.Ay, label=L"\mathrm{Raw}")

# create inset axis
axins = ax.inset_axes([0.475, 0.5, 0.5, 0.45]) # [x, y, width, height]
axins.plot(df.Ax, df.Ay, label=L"\mathrm{Raw}")
axins.set_xlim(0, 3)
axins.set_ylim(-35, -5)

# draw a rectangle around the inset axis
ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

ax.set_xlabel(L"f\ (\mathrm{kHz})")
ax.set_ylabel(L"A\ (\mathrm{dBV})")
ax.grid()

# ax[0].set_ylim(-35, -5)
# plt.legend(loc="lower left")
plt.tight_layout()
# plt.savefig(joinpath(@__DIR__, "bilder/ledfft.pdf"), bbox_inches="tight")
plt.show()
end

df = CSV.read(joinpath(@__DIR__, "data/1kfft.trc"), DataFrame, header=["Ax", "Ay"], skipto=17)
df2 = CSV.read(joinpath(@__DIR__, "data/100kfft.trc"), DataFrame, header=["Ax", "Ay"], skipto=17)
df3 = CSV.read(joinpath(@__DIR__, "data/1Mfft.trc"), DataFrame, header=["Ax", "Ay"], skipto=17)
df.Ax /= 1000
df2.Ax /= 1000
df3.Ax /= 1000
# plot ax vs Ay
begin
fig, ax = plt.subplots(2, 1, figsize=(7.5, 5.3))
for axis in ax
    axis.plot(df.Ax, df.Ay, lw=0.1)
    axis.plot(df2.Ax, df2.Ay, lw=0.1)
    axis.plot(df3.Ax, df3.Ay, lw=0.1)
    axis.set_xlabel(L"f\ (\mathrm{kHz})")
    axis.set_ylabel(L"A\ (\mathrm{dBV})")
    axis.grid()
end
ax[1].set_ylim(-55, -18)
ax[1].set_xlim(0, 150)

# dummyplot for legend
ax[1].plot([], [], label=L"1\ \mathrm{k}\Omega", c="C0")
ax[1].plot([], [], label=L"100\ \mathrm{k}\Omega", c="C1")
ax[1].plot([], [], label=L"1\ \mathrm{M}\Omega", c="C2")

# change axis padding
plt.legend(borderaxespad=0.5)
plt.tight_layout()

draw_insert_from(fig, ax[0], ax[1], lines_to_draw=[1, 1, 0, 0])

# plt.savefig(joinpath(@__DIR__, "bilder/Rfft.pdf"), bbox_inches="tight")
plt.show()
end