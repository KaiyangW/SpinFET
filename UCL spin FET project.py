import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.06e-34
h = 6.63e-34
q = 1.6e-19
e = np.e
G_QPC = 2 * (e**2) / h  # QPC quantized conductance
kf = 1.6e8  # Fermi wave vector [m^-1]
Width = 8 * 1e-6  # width=8um
m_eff = 0.04 * 9.1e-31  # electron effective mass [kg]
LCH = 2e-6  # [m] - Channel Length
LCH2 = 4e-6
lambda_ = 2 * np.pi / kf  # λ - wavelength
M = 2 * Width / lambda_
G0 = M * G_QPC
g1 = 9  # g-factor for InGaAs
g2 = 1.93  # g-factor for ZnO/ZnMgO
g3 = 4.5  # g-factor for LaAlO3/SrTiO3
u_Bohr = 9.27e-24  # Bohr magneton
B_ext = 0.12  # [T]

NN = 500

rso12 = np.zeros((4, 4, NN))
rso21 = np.zeros_like(rso12)

det_X = np.zeros(NN)
det_Y = np.zeros(NN)
det_X2 = np.zeros(NN)
det_Y2 = np.zeros(NN)
det_X3 = np.zeros(NN)
det_Y3 = np.zeros(NN)

def G_spinFET(G0, rashba_alpha, m_eff, LCH, kx):
    # Dummy implementation for G_spinFET
    return np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))

for index in range(NN):
    al = 20e-12 - index * 2.2e-14
    alpha = q * np.linspace(0, al, NN)  # |α| ∼ 20 × E−12 eVm [Chuang]
    rashba_alpha = alpha[index]
    NS = 2000
    phi_space = np.linspace(-np.pi / 2, np.pi / 2, NS)
    ky_space = kf * np.sin(np.arctan(phi_space))
    kx_space = kf * np.cos(np.arctan(phi_space))
    ds = phi_space[1] - phi_space[0]  # N+1-N

    # Circuit model
    Period = 0
    for ss in range(NS):
        kx = kx_space[ss]
        # Conductance
        rso11, rso12[:,:,index], rso21[:,:,index], rso22 = G_spinFET(G0, rashba_alpha, m_eff, LCH, kx)
        Period += ds

    # Transmission model
    det_x = np.zeros(len(ky_space))
    det_y = np.zeros(len(ky_space))
    det_x2 = np.zeros_like(det_x)
    det_y2 = np.zeros_like(det_y)
    det_x3 = np.zeros_like(det_x)
    det_y3 = np.zeros_like(det_y)

    for iky in range(len(ky_space)):
        ky = ky_space[iky]
        kx = np.sqrt(kf**2 - ky**2)
        theta = 2 * m_eff * rashba_alpha * LCH / hbar**2 - g1 * u_Bohr * B_ext * m_eff * LCH / (hbar**2 * kx)
        theta2 = 2 * m_eff * rashba_alpha * LCH / hbar**2 - g2 * u_Bohr * B_ext * m_eff * LCH / (hbar**2 * kx)
        theta3 = 2 * m_eff * rashba_alpha * LCH / hbar**2 - g3 * u_Bohr * B_ext * m_eff * LCH / (hbar**2 * kx)
        s = ky / kf
        det_x[iky] = 3 * s**2 + 2 * (1 - s**2) * np.cos(theta)
        det_y[iky] = (1 - s**2) + s**2 * np.cos(theta / np.sqrt(1 - s**2)) * np.sin(np.sqrt(1 - s**2))
        det_x2[iky] = 3 * s**2 + 2 * (1 - s**2) * np.cos(theta2)
        det_y2[iky] = (1 - s**2) + s**2 * np.cos(theta2 / np.sqrt(1 - s**2)) * np.sin(np.sqrt(1 - s**2))
        det_x3[iky] = 3 * s**2 + 2 * (1 - s**2) * np.cos(theta3)
        det_y3[iky] = (1 - s**2) + s**2 * np.cos(theta3 / np.sqrt(1 - s**2)) * np.sin(np.sqrt(1 - s**2))

    # Normalize: sum over ky
    det_X[index] = np.sum(det_x) / len(kx_space) / Period
    det_Y[index] = np.sum(det_y) / Period / len(ky_space)
    det_X2[index] = np.sum(det_x2) / len(kx_space) / Period
    det_Y2[index] = np.sum(det_y2) / Period / len(ky_space)
    det_X3[index] = np.sum(det_x3) / len(kx_space) / Period
    det_Y3[index] = np.sum(det_y3) / Period / len(ky_space)

rso12 /= Period
rso21 /= Period

Modx = rso21[2, 2, :]
Mody = rso12[3, 3, :]
Sum_det = np.sqrt(det_X**2 + det_Y**2)
Sum_det2 = np.sqrt(det_X2**2 + det_Y2**2)
Sum_det3 = np.sqrt(det_X3**2 + det_Y3**2)

ras_coefit = alpha / q / 1e-12

plt.figure(1)  # Transmission model plot
plt.plot(ras_coefit / 10, det_X, 'b', linewidth=1)
plt.plot(ras_coefit / 10, det_X2, 'k', linewidth=1)
plt.plot(ras_coefit / 10, det_X3, 'r', linewidth=1)
plt.plot(ras_coefit / 10, det_Y, 'b', linewidth=1)
plt.plot(ras_coefit / 10, det_Y2, 'k', linewidth=1)
plt.plot(ras_coefit / 10, det_Y3, 'r', linewidth=1)
plt.plot(ras_coefit / 10, Sum_det, 'b', linewidth=1)
plt.plot(ras_coefit / 10, Sum_det2, 'k', linewidth=1)
plt.plot(ras_coefit / 10, Sum_det3, 'r', linewidth=1)
plt.legend(['g=9', 'g=1.95', 'g=4.5'])
plt.xlabel('RSO α (10^{-12} eV-m)')
plt.ylabel('Detector voltage V')

# Ensure Modx and Mody are not zero to avoid division by zero errors
if Modx[-1] != 0 and Mody[-1] != 0:
    plt.figure(2)  # Circuit model plot
    plt.plot(ras_coefit / 10, Modx / Modx[-1] * det_X2[-1], 'k', linewidth=1)
    plt.plot(ras_coefit / 10, Mody / Mody[-1] * det_Y2[-1], 'b', linewidth=1)
    plt.legend(['Circuit model (X)', 'Circuit model (Y)'])
    plt.xlabel('RSO α (10^{-12} eV-m)')
    plt.ylabel('Detector voltage V')

    plt.figure(3)  # Compare
    plt.plot(ras_coefit / 10, Modx / Modx[-1] * det_X2[-1], 'k', linewidth=1)
    plt.plot(ras_coefit / 10, det_X2, 'b', linewidth=1)
    plt.legend(['Circuit model', 'Transmission model'])
    plt.xlabel('RSO α (10^{-12} eV-m)')
    plt.ylabel('Detector voltage V')

# QPC NEGF model
unit = 5e-10
Const = hbar**2 / (2 * m_eff * unit**2 * q)
Nm = 25
Nn = 25
Hd = np.zeros((Nm * Nn, Nm * Nn))

for index_m in range(1, Nm+1):
    for index_n in range(1, Nn+1):
        i1 = index_m + (index_n - 1) * Nm - 1
        V0 = 0
        x = unit * (index_n + (index_m - 1) * Nn)
        y = x
        Ux = 0.027
        Uy = 0.04
        poteV = V0 - Ux * x**2 + Uy * y**2
        Hd[i1, i1] = 4 * Const + poteV

        if index_n > 1:
            Hd[i1, index_n - 1 + (index_m - 1) * Nn - 1] = -Const
        if index_n < Nn:
            Hd[i1, index_n + 1 + (index_m - 1) * Nn - 1] = -Const
        if index_m > 1:
            Hd[i1, index_n + (index_m - 2) * Nn - 1] = -Const
        if index_m < Nm:
            Hd[i1, index_n + index_m * Nn - 1] = -Const

counter = 1
sig = -1j * np.eye(25)
Trans = []
energy = []

def sigma_function(E, sig, Nm, Nn, Const):
    # Dummy implementation for sigma_function
    return sig

G_con = []

for E in np.arange(0, 0.501, 0.001):
    sig = -1j * np.eye(25)
    sig = sigma_function(E, sig, Nm, Nn, Const)
    Sig1 = np.zeros_like(Hd)
    Sig2 = np.zeros_like(Hd)
    Sig1[:Nm, :Nm] = np.real(sig)
    Sig2[-Nm:, -Nm:] = np.real(sig)
    Gam1 = 1j * (Sig1 - Sig1.T)
    Gam2 = 1j * (Sig2 - Sig2.T)
    Gd = np.linalg.inv(E * np.eye(Nm * Nn) - Hd - Sig1 - Sig2)
    Trans.append(np.trace(Gd @ Gam1 @ Gd.T @ Gam2) / 1e-12)
    G_con.append(np.abs(Trans[-1]) * (2 * e**2 / h) / 1e47)
    energy.append(E)
    counter += 1

plt.figure(4)
plt.plot(energy, G_con, 'k')
plt.xlabel('Energy (eV)')
plt.ylabel('G (2e^2/h)')
plt.title('Transmission')

plt.show()
