from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


from scipy.spatial import KDTree, cKDTree
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, interp1d
from scipy.integrate import quad

from collections import Counter
from pycorr import TwoPointCorrelationFunction

import psutil
import multiprocessing as mp

import re
import asdf


## DATA

def extract_redshift_from_path(file_path):
    # Look for a similar pattern as "zX.XXX" in the path
    match = re.search(r'z(\d+\.\d+)', file_path)
    if match:
        # Extract redshif value
        return float(match.group(1))
    else:
        raise ValueError("Redshift (z) not found in the file path.")
    
def get_box_size(file_path):
    box_size = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("BOX_SIZE"):
                # Extract value after the "=" sign
                box_size = int(line.split('=')[1].strip())
                break
    return box_size

def get_variable(file_path, name):
    var = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(f'{name}'):
                value = line.split('=')[1].strip()
                try:
                    # try to convert as int
                    var = int(value)
                except ValueError:
                    try:
                        # try to convert to float
                        var = float(value)
                    except ValueError:
                        # keep the var as a text chain
                        var = value
                break
    return var

def extract_positions_from_files(file_list):
    """
    Extracts particle positions from a list of ASDF files.
    
    Parameters:
        file_list (list of str): Liste des chemins vers les fichiers ASDF.
    
    Returns:
        np.ndarray: Un tableau numpy contenant les positions concaténées des particules.
    """
    positions = []
    for file in file_list:
        with asdf.open(file) as af:
            if 'pos' in af:  # Check if 'pos' is in the file
                positions.append(af['pos'])
            else:
                print(f"'pos' key not found in {file}. Skipping...")
    return np.vstack(positions)

## MASS 

def print_sample_mass_distribution(mass_samples, labels, title, colors=None):
    plt.figure(figsize=(7, 6))
    
    for i, masses in enumerate(mass_samples):
        counts, bins = np.histogram(masses, bins=100)
        plt.loglog(bins[:-1], counts, marker='o', linestyle='none', markersize=3, 
                   label=labels[i], color=colors[i])

    plt.xlabel("Halo Mass (M☉)")
    plt.ylabel("Number of halos")
    plt.title(f"Halo Mass Distribution, {title}")
    plt.legend()
    plt.show()

def filter_mass(data, threshold):
    return data[data >= threshold]

def compute_mass_sums(data, bins):
    mass_sums = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        if i < len(bins) - 2:
            mask = (data >= bins[i]) & (data < bins[i + 1])
        else:
            mask = (data >= bins[i]) & (data <= bins[i + 1])
        mass_sums[i] = np.sum(data[mask])
    return mass_sums

def print_total_mass(masses, bins, labels, colors, title):
    plt.figure(figsize=(7, 6))
    
    for i, masses in enumerate(masses):
        plt.loglog(bins[:-1], masses, marker='o', linestyle='none', markersize=3, 
                   label=labels[i], color=colors[i])

    plt.xlabel("Halo Mass (M☉)")
    plt.ylabel("Total mass of the bin (M☉)")
    plt.title(f"Total mass per bin, {title}")
    plt.legend()
    plt.show()

def compute_hmf_vol(halo_masses, bins, box_size):
    """
    Calcule la HMF classique : nombre de halos par unité de masse et par unité de volume.
    
    Args:
        halo_masses (array): Tableau des masses des halos.
        bins (array): Tableau des bords des bins de masse (logarithmiques).
        box_size (float): Taille de la boîte de simulation (Mpc/h).
    
    Returns:
        hmf (array): HMF (nombre de halos par unité de masse et de volume).
        bin_centers (array): Centres des bins de masse.
    """
    # Counting the number of halos in each bins 
    counts, bin_edges = np.histogram(halo_masses, bins=bins)
    
    # Mass bins width
    bin_widths = np.diff(bins)
    
    # Box volume (in Mpc³/h³)
    volume = box_size**3
    
    # HMF computation : number of halos per mass and volume unit 
    hmf = counts / (bin_widths * volume)
    
    # Bins centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    return hmf, bin_centers

def compute_hmf(halo_masses, bins):

    # Count the number of halos in each bins
    counts, bin_edges = np.histogram(halo_masses, bins=bins)
    
    # Width of mass bins
    # bin_widths = np.diff(bins)
    
    # HMF : number of halos per mass unit
    hmf = counts # / bin_widths
    
    # Bins centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    return hmf, bin_centers

def xi_wrt_mass(mass_bins, sampled_masses, positions, bins_rp, bins_pi, boxsize, save_path, mask_range=(None, None), nthreads=32):

    # Store the results of r and xi for each mass bin
    wp_all = []
    rp_all = []
    # rppi_pi_all = []

    for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):

        # Selecting halos in this mass bin
        mask = (sampled_masses >= low) & (sampled_masses < high)
        halos_in_bin = positions[mask]

        # Check for halos in the bin
        if halos_in_bin.shape[0] == 0:
            print(f"Skipping mass bin {i} because it contains no halos.")
            continue

        print(f"Mass bin {i}: {halos_in_bin.shape[0]} halos")

        # Separate x, y, z if necessary
        if halos_in_bin.shape[1] == 3:
            halos_in_bin = np.array([halos_in_bin[:, 0], halos_in_bin[:, 1], halos_in_bin[:, 2]])

        print("Nombre de halos dans le bin :", halos_in_bin.shape[1])
        print("Nombre de bins rp:", len(bins_rp))
        print("Nombre de bins pi:", len(bins_pi))

        print(f"Nombre total de halos dans le bin {i}: {halos_in_bin.shape[1]}")
        print(f"Nombre de bins_rp: {len(bins_rp)}, Nombre de bins_pi: {len(bins_pi)}")
        print(f"Min bins_rp: {np.min(bins_rp)}, Max bins_rp: {np.max(bins_rp)}")

        # Calculation of the halo-halo correlation function in rppi mode
        results_rppi = TwoPointCorrelationFunction(
            mode='rppi',
            edges=(bins_rp, bins_pi), 
            data_positions1=halos_in_bin,
            boxsize=boxsize,
            nthreads=nthreads,
            los='z'
        )
        
        rp, wp = results_rppi(pimax=None, return_sep=True)
        
                
        print("Type de rp:", type(rp))
        print("Shape de rp:", rp.shape)
        
        print(np.isnan(rp).sum())
        print(np.isnan(wp).sum())

        print("Min rp:", np.nanmin(rp))
        print("Max rp:", np.nanmax(rp))
        print("Min bins_rp:", np.min(bins_rp))
        print("Max bins_rp:", np.max(bins_rp))

        # Applying the mask by r BEFORE flattening
        if mask_range == (None, None):
            mask = np.ones_like(rp, dtype=bool)
        else:
            mask = (rp >= mask_range[0]) & (rp < mask_range[1])

        print(np.sum(mask))

        # Applying the mask in 2D
        rp_masked = rp[mask]
        wp_masked = wp[mask]

        # # Flatten the masked arrays
        # rp_flat = rp_masked.flatten()
        # pi_flat = pi_masked.flatten()
        # xi_rppi_flat = xi_rppi_masked.flatten()

        # Storing the rppi results for this mass bin
        rp_all.append(rp_masked)
        # rppi_pi_all.append(pi_flat)
        wp_all.append(wp_masked)

    # Saving results
    np.savez(save_path, 
             mass_bins=mass_bins, 
             rp_all=rp_all, 
             wp_all=wp_all) #             rppi_pi_all=rppi_pi_all, 

    print(f"\n Fichier enregistré : {save_path}")

    return wp_all, rp_all #, rppi_xi_hh_all


## STATS

def sample_take(id, mass, percent):

    data = {'id': id, 'mass': mass}

    df = pd.DataFrame(data)
    df_sample = df[np.random.rand(len(df)) < percent]

    sampled_indices = df_sample.index.tolist()

    return sampled_indices

def compute_differences(mass_sums1, mass_sums2, epsilon=1e-10):
    relative_diff = (mass_sums1 - mass_sums2) / (mass_sums2 + epsilon)
    absolute_diff = mass_sums1 - mass_sums2
    return relative_diff, absolute_diff

def calculate_stats(data, label, verbose=True):
    mean = np.mean(data)
    std = np.std(data)
    if verbose :
        print(f'{label} - Mean mass : {mean:.2e}, Standard deviation : {std:.2e}')
    return mean, std

def normalize_distribution(mass_sums):
    return mass_sums / np.sum(mass_sums)

def ratio_counts(count1, count2, epsilon =1e-10):
    ratio_counts = count1 / (count2 + epsilon)
    return ratio_counts

## PLOTS

def plot_loglog_bins(data, bins, colors, labels, title, yname):
    """Plot relative differences for wanted distributions."""
    n_plots = len(data)
    fig, axes = plt.subplots(1, n_plots, sharex=True, figsize=(5 * n_plots, 5))
    
    # Handle case of a single plot (axes is not an iterable)
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(f'{title}')
    
    for i, (ax, diff, bin_edges, color, label) in enumerate(zip(axes, data, bins, colors, labels)):
        ax.loglog(bin_edges[:-1], diff, marker='o', linestyle='none', markersize=3, color=color)
        ax.set_title(label)
        ax.set(ylabel=f'{yname}', xlabel=r'Halo Mass $[h^{-1} \,M☉]$')
    
    fig.tight_layout()
    plt.show()

def rsd_effect(a, positions, velocities, Hz, name):
    """
    Visualisation of RSD effect of positions
    """

    print("Shape of velocities:", velocities.shape)
    print("Shape of positions:", positions.shape, '\n')

    vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
    x, y, z = positions[:,0], positions[:,1], positions[:,2]
    # print('Velocities: ', vx, vy, vz, '\n')  # km/s

    print("Hubble constant Hz:", Hz, '\n')

    # Computation of RSD position
    z_rsd = z + vz / (a * Hz)  # Mpc/h

    # print('z:', z, '\n z rsd:', z_rsd, '\n')

    # Computation of difference delta_z
    delta_z = z_rsd - z
    # print('delta z :', delta_z)

    # Plot
    # plt.figure(figsize=(6, 5))
    # plt.scatter(z, z_rsd, alpha=0.5, s=10, c='blue')
    # plt.plot(z, z, color='red', linestyle='--', label=r'$z = z_{\mathrm{rsd}}$')
    # plt.xlabel(r"$z$ (Position)", fontsize=14)
    # plt.ylabel(r"$z_{\mathrm{rsd}}$", fontsize=14)
    # plt.xscale('linear')
    # plt.title(f"RSD effect on position, {name}", fontsize=16)
    # plt.grid(alpha=0.3)
    # plt.legend(fontsize=12, loc='upper left')
    # plt.show()

    return z_rsd, delta_z

def check_empty_bins(datas, labels, title):
    bins_empty_list = []
    print(f'\n {title} : \n')
    for data, label in zip(datas, labels):
        bins_empty = np.where(data == 0)[0]
        bins_empty_list.append(bins_empty)
        print(f"{label} : {bins_empty}")
    return bins_empty_list

def print_multipoles(results_smu, ells, title ): #wedges if necessary

    fig, lax = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False, figsize=(12, 5))
    fig.suptitle(rf'2PCF evolution across Multipoles - {title}') # And angular Intervals - if wedges

    # Let us project to multipoles (monopole, quadruple, hexadecapole)
    s, xiell = results_smu(ells=ells, return_sep=True)
    ax = lax[0]
    for ill, ell in enumerate(ells):
        ax.plot(s, s**2 * xiell[ill], label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$')
    ax.set_title('Large-scale evolution')

    # Let us project to wedges
    # s, ximu = results_smu(wedges=wedges, return_sep=True)
    ax = lax[1]
    for ill, ell in enumerate(ells):
        ax.plot(s, s * xiell[ill], label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s \xi_{\ell}(s)$')
    ax.set_title('Zoom on small-scale')
    plt.show()

def print_many_multipoles(results_list, labels, colors_list, ells, title): #wedges if necessary

    fig, lax = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False, figsize=(12, 5))
    fig.suptitle(rf'2PCF evolution across Multipoles - {title}') # And angular Intervals - if wedges

    # Let us project to multipoles (monopole, quadruple, hexadecapole)
    ax = lax[0]
    color_idx = 0
    for idx, results_smu in enumerate(results_list):
        s, xiell = results_smu(ells=ells, return_sep=True)
        for ill, ell in enumerate(ells):
            ax.plot(s, s**2 * xiell[ill], label=rf'{labels[idx]}, $\ell = {ell}$', color=colors_list[color_idx])
            color_idx +=1
    
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$')
    ax.set_title('Large-scale evolution')

    # Let us project to wedges
    # s, ximu = results_smu(wedges=wedges, return_sep=True)
    ax = lax[1]
    color_idx = 0
    for idx, results_smu in enumerate(results_list):
        s, xiell = results_smu(ells=ells, return_sep=True)
        for ill, ell in enumerate(ells):
            ax.plot(s, s * xiell[ill], label=rf'{labels[idx]}, $\ell = {ell}$', color=colors_list[color_idx])
            color_idx +=1
    
    ax.legend(ncol=2)
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s \xi_{\ell}(s)$')
    ax.set_title('Zoom on small-scale')
    plt.show()

def print_multipoles_log(results_smu, ells, title ): #wedges if necessary

    fig, lax = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False, figsize=(12, 5))
    fig.suptitle(rf'2PCF evolution across Multipoles - {title}')

    # Let us project to multipoles (monopole, quadruple, hexadecapole)
    s, xiell = results_smu(ells=ells, return_sep=True)
    ax = lax[0]
    for ill, ell in enumerate(ells):
        ax.semilogx(s, s**2 * xiell[ill], label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$')

    # Let us project to wedges
    ax = lax[1]
    for ill, ell in enumerate(ells):
        ax.semilogx(s, s * xiell[ill], label=r'$\ell = {:d}$'.format(ell))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s \xi_{\ell}(s)$')
    plt.show()

def print_many_multipoles_log(results_list, labels, colors_list, ells, title): #wedges if necessary

    fig, lax = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False, figsize=(12, 5))
    fig.suptitle(rf'2PCF evolution across Multipoles - {title}')

    # Let us project to multipoles (monopole, quadruple, hexadecapole)
    ax = lax[0]
    color_idx = 0
    for idx, results_smu in enumerate(results_list):
        s, xiell = results_smu(ells=ells, return_sep=True)
        for ill, ell in enumerate(ells):
            ax.semilogx(s, s**2 * xiell[ill], label=rf'{labels[idx]}, $\ell = {ell}$', color=colors_list[color_idx])
            color_idx +=1

    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$')

    # Let us project to wedges
    ax = lax[1]
    color_idx = 0
    for idx, results_smu in enumerate(results_list):
        s, xiell = results_smu(ells=ells, return_sep=True)
        for ill, ell in enumerate(ells):
            ax.semilogx(s, s * xiell[ill], label=rf'{labels[idx]}, $\ell = {ell}$', color=colors_list[color_idx])
            color_idx +=1
    
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s \xi_{\ell}(s)$')
    plt.show()

def print_chosen_multipole(results_r, results_xi, legends, title, colors_list, scalex=None, scaley=None, yrange=None, mask_range=None):

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'2PCF Comparison of {title}')

    # Projection to monopole
    
    for r, xi, legend, color in zip(results_r, results_xi, legends, colors_list):
        ax.plot(r, r * r * xi.squeeze(), label=legend, linestyle="-", color = color)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$')
    if scalex =='log':
        ax.set_xscale('log')

    if scaley =='log':
        ax.set_yscale('log')

    # If `yrange` is defined, fix Y axis limits
    if yrange is not None:
        plt.ylim(yrange)

    if mask_range is not None:
        plt.xlim(mask_range)

    plt.show()

def plot_masses_fct_or(data_list, bins_list, colors_list, labels_list, linestyles_list, ylabel, title, scale='loglog', grid=True, figsize=(7, 5)):
    
    plt.figure(figsize=figsize)
    
    for bins, data, label, color, linestyle in zip(bins_list, data_list, labels_list, colors_list, linestyles_list):
        if scale == "loglog":
            plt.loglog(bins, data, label=label, color=color, marker='o', linestyle=linestyle, markersize=3)
        elif scale == "semilogx":
            plt.semilogx(bins, data, label=label, color=color, marker='o', linestyle=linestyle, markersize=3)
        elif scale == "semilogy":
            plt.semilogy(bins, data, label=label, color=color, marker='o', linestyle=linestyle, markersize=3)
        else:
            raise ValueError("Paramètre 'scale' invalide. Utiliser 'loglog', 'semilogx' ou 'semilogy'.")

    plt.xlabel(r'Halo Mass $[h^{-1} \,M☉]$')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    if grid:
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    plt.show()


def plot_masses_fct(data_list, bins_list, colors_list, labels_list, 
                    linestyles_list, ylabel, title, scale='loglog', 
                    grid=True, figsize=(7, 5), margin_factor=0.05, 
                    errors_list=None, center_around_one=False):
    
    plt.figure(figsize=figsize)
    
    all_y_values = []  # Stocke toutes les valeurs y pour ajuster l'échelle

    for bins, data, label, color, linestyle, errors in zip(
            bins_list, data_list, labels_list, colors_list, linestyles_list, 
            errors_list if errors_list else [None] * len(data_list)
        ):

        if errors is not None:
            # Plot with error bars
            plt.errorbar(bins, data, yerr=errors, fmt='o-', label=label, 
                         color=color, linestyle=linestyle, markersize=3, 
                         capsize=5, elinewidth=1)
        else:
            plt.plot(bins, data, label=label, color=color, marker='o', 
                     linestyle=linestyle, markersize=3)
        
        all_y_values.extend(data)  # Stocke les valeurs y

    # Définir les échelles des axes
    if scale == "semilogx":
        plt.xscale("log")
    elif scale == "loglog":
        plt.xscale("log")
        plt.yscale("log")
    elif scale == "semilogy":
        plt.yscale("log")
    else:
        raise ValueError("Paramètre 'scale' invalide. Utiliser 'loglog', 'semilogx' ou 'semilogy'.")

    plt.xlabel(r'Halo Mass $[h^{-1} \,M☉]$')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Centrer autour de y=1 seulement si demandé
    if center_around_one:
        y_min, y_max = np.nanmin(all_y_values), np.nanmax(all_y_values)  # Ignorer les NaN
        max_dev = max(abs(y_max - 1), abs(1 - y_min))  # Distance max par rapport à 1

        # Ajouter une marge pour éviter que les points touchent les bords
        margin = margin_factor * max_dev  
        new_y_min = 1 - max_dev - margin
        new_y_max = 1 + max_dev + margin

        # Centrer autour de y=1 avec une marge
        plt.ylim(new_y_min, new_y_max)  
        
        # Ajouter la ligne de référence y=1
        plt.axhline(1, color="black", linestyle="--", linewidth=1)  

    if grid:
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)

    plt.show()

def projected_pcf(results_rppi, title):

    sep, wp = results_rppi(pimax=None, return_sep=True) # sep is r_p

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False, figsize=(13, 5))
    fig.suptitle(f'Projected Correlation Function, {title}')

    ax1.plot(sep, sep * wp)
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$r_{p}$')
    ax1.set_ylabel(r'$r_{p} w_{p}(r_{p})$')
    ax1.grid(True)

    ax2.plot(sep, wp)
    ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.set_xlabel(r'$r_{p}$')
    ax2.set_ylabel(r'$w_{p}$')
    ax2.grid(True)

    plt.show()
    return sep

def print_xi(results_rppi, pi_lim, r_lim, title):

    sep, wp = results_rppi(pimax=None, return_sep=True)
    pi = results_rppi.sepavg(axis=1)
    xi_rppi = results_rppi.corr

    #  MASQUES
    mask_pi = (pi >= -pi_lim) & (pi <= pi_lim)
    mask_rp = (sep > r_lim)

    sep_filtered = sep[mask_rp]
    pi_filtered = pi[mask_pi]

    xi_rppi_filtered = xi_rppi[mask_rp, :]
    xi_rppi_filtered = xi_rppi_filtered[:, mask_pi]

    # Apply log10 only to positiv values
    xi_rppi_log = np.sign(xi_rppi_filtered) * np.log10(np.abs(xi_rppi_filtered) + 1e-5) - np.sign(xi_rppi_filtered) * np.log10(1e-5)
    # min_xi = np.min(xi_rppi_log)
    # print("Minimum de xi_rppi_log:", min_xi)

    RP, PI = np.meshgrid(sep_filtered, pi_filtered)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(RP, PI, xi_rppi_log.T, levels=45, cmap='coolwarm')  # contourf for the 2D correlation function

    cbar = plt.colorbar(cp)
    cbar.set_label(r'$\xi(r_p, \pi)$')

    plt.xlabel(r'$r_p \, [h^{-1} \, Mpc]$')
    plt.ylabel(r'$\pi \, [h^{-1} \, Mpc]$')
    plt.title(f'2D Correlation Function $\\xi(r_p, \\pi)$, {title}')

    contour_levels = np.array([-1, -0.1, 0.1, 1, 2, 3, 4])
    valid_levels = contour_levels[np.isfinite(contour_levels)]
    contours = plt.contour(RP, PI, xi_rppi_log.T, levels=valid_levels, colors='black', linewidths=1)

    plt.clabel(contours, inline=True, fontsize=7, fmt='%1.2f')
    plt.grid(True)
    plt.show()

def print_xi_round(results_rppi, pi_lim, r_lim, r_lim_min, title):
    """
    Affiche la fonction de corrélation 2D avec un miroir des valeurs de r_p (partie négative ajoutée).
    """

    sep, wp = results_rppi(pimax=None, return_sep=True)  # r_p
    pi = results_rppi.sepavg(axis=1)  # pi
    xi_rppi = results_rppi.corr  # Corrélation

    #  MASQUES
    mask_pi = (pi >= -pi_lim) & (pi <= pi_lim)
    mask_rp = (sep > r_lim_min) & (sep <= r_lim)

    sep_filtered = sep[mask_rp]
    pi_filtered = pi[mask_pi]

    xi_rppi_filtered = xi_rppi[mask_rp, :]
    xi_rppi_filtered = xi_rppi_filtered[:, mask_pi]

    #  Étendre r_p en prenant le miroir (symétrie)
    sep_sym = np.concatenate([-sep_filtered[::-1], sep_filtered])  

    #  Appliquer log10 uniquement aux valeurs positives
    xi_rppi_log_filtered = np.sign(xi_rppi_filtered) * np.log10(np.abs(xi_rppi_filtered) + 1e-5) - np.sign(xi_rppi_filtered) * np.log10(1e-5)

    #  Étendre xi_rppi pour refléter cette symétrie
    xi_rppi_sym = np.concatenate([xi_rppi_log_filtered[::-1, :], xi_rppi_log_filtered], axis=0)

    # min_xi = np.min(xi_rppi_log_filtered)
    # print("Minimum de xi_rppi_log_filtered:", min_xi)

    #  Correction du meshgrid
    RP, PI = np.meshgrid(sep_sym, pi_filtered)

    # Création du graphique
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(RP, PI, xi_rppi_sym.T, levels=45, cmap='coolwarm')

    cbar = plt.colorbar(cp)
    cbar.set_label(r'$\xi(r_p, \pi)$')

    plt.xlabel(r'$r_p \, [h^{-1} \, Mpc]$')
    plt.ylabel(r'$\pi \, [h^{-1} \, Mpc]$')
    plt.title(f'2D Correlation Function $\\xi(r_p, \\pi)$, {title}')

    #  Contours en noir pour certaines valeurs clés
    contour_levels = [-2, -1.5, -0.1, 1, 2, 3, 4] #, 0.1, 1
    contours = plt.contour(RP, PI, xi_rppi_sym.T, levels=contour_levels, colors='black', linewidths=1)
    plt.clabel(contours, inline=True, fontsize=7, fmt='%1.2f')

    plt.grid(True)
    plt.show()

def print_xi_round_test_nolog(results_rppi, pi_lim, r_lim, r_lim_min, title):
    """
    Affiche la fonction de corrélation 2D avec un miroir des valeurs de r_p (partie négative ajoutée).
    """

    sep, wp = results_rppi(pimax=None, return_sep=True)  # r_p
    pi = results_rppi.sepavg(axis=1)  # pi
    xi_rppi = results_rppi.corr  # Corrélation

    #  MASQUES
    mask_pi = (pi >= -pi_lim) & (pi <= pi_lim)
    mask_rp = (sep > r_lim_min) & (sep <= r_lim)

    sep_filtered = sep[mask_rp]
    pi_filtered = pi[mask_pi]

    xi_rppi_filtered = xi_rppi[mask_rp, :]

    #  Étendre r_p en prenant le miroir (symétrie)
    sep_sym = np.concatenate([-sep_filtered[::-1], sep_filtered])  

    #  Étendre xi_rppi pour refléter cette symétrie
    xi_rppi_sym = np.concatenate([xi_rppi_filtered[::-1, :], xi_rppi_filtered], axis=0)

    # #  Appliquer log10 uniquement aux valeurs positives
    # xi_rppi_log = np.sign(xi_rppi_sym) * np.log10(np.abs(xi_rppi_sym) + 1e-5)

    # #  Appliquer ce masque aux matrices xi_rppi
    # xi_rppi_log_filtered = xi_rppi_log[:, mask_pi]

    #  Correction du meshgrid
    RP, PI = np.meshgrid(sep_sym, pi_filtered)

    # Création du graphique
    plt.figure(figsize=(8, 6))
    linthresh = 0.1  # Seuil en-dessous duquel la colormap devient linéaire (évite les problèmes avec 0)
    norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=0.1, vmin=-np.max(np.abs(xi_rppi_sym)), vmax=np.max(np.abs(xi_rppi_sym)))
    cp = plt.contourf(RP, PI, xi_rppi_sym[:, mask_pi].T, levels=45, cmap='Blues', norm=norm)

    cbar = plt.colorbar(cp)
    cbar.set_label(r'$\xi(r_p, \pi)$')

    plt.xlabel(r'$r_p \, [h^{-1} \, Mpc]$')
    plt.ylabel(r'$\pi \, [h^{-1} \, Mpc]$')
    plt.title(f'2D Correlation Function $\\xi(r_p, \\pi)$, {title}')

    #  Contours en noir pour certaines valeurs clés
    contour_levels = [1, 1.5, 2, 3, 4] #, 0.1, 1
    contours = plt.contour(RP, PI, xi_rppi_sym[:, mask_pi].T, levels=contour_levels, colors='black', linewidths=1)
    plt.clabel(contours, inline=True, fontsize=7, fmt='%1.2f')

    plt.grid(True)
    plt.show()

def compa_pcf(results_rppi_list, legends, colors, title, mask_range=(0.01, 40)):

    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=16)

    for i, results_rppi in enumerate(results_rppi_list):
        # Get r_p and w_p(r_p) values
        sep, wp = results_rppi(pimax=None, return_sep=True)  # sep = r_p

        # Filter between mask ranges in Mpc/h
        mask = (sep >= mask_range[0]) & (sep <= mask_range[1])
        sep_filtered = sep[mask]
        wp_filtered = wp[mask]

        # Plot with a distinct style
        plt.loglog(sep_filtered, sep_filtered * wp_filtered, label=legends[i], linestyle="-", color=colors[i])

    # Axes et légende
    plt.xlabel(r"$r_{p}$ [Mpc/h]", fontsize=14)
    plt.ylabel(r"$r_{p} w_{p}(r_{p})$", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.show()

def plot_mass_binned_quantity(r_hh_all, quantity_all, mass_bins, ylabel, title, mask_range=None, apply_transformation=None, grid=True, yrange = None, figsize=(7, 5), scalex='log', scaley='log'):

    plt.figure(figsize=figsize)
    n_bins = len(mass_bins) - 1  # Number of bins

    for i in range(n_bins):
        r_hh = r_hh_all[i]  # Get r for this bin
        quantity = quantity_all[i]  # Get associated value

        # Apply a transformation if necessary (ex: r^2 * ξ_0(r))
        if apply_transformation is not None:
            quantity = apply_transformation(r_hh, quantity)

        # Apply a mask if necessary
        if mask_range is not None:
            mask = (r_hh >= mask_range[0]) & (r_hh < mask_range[1])
            r_hh = r_hh[mask]
            quantity = quantity[mask]

        plt.plot(r_hh, quantity, label=f'Mass bin {i}', linestyle='-')

    if scalex =='log':
        plt.xscale('log')

    if scaley =='log':
        plt.yscale('log')

    plt.xlabel(r'$r \, [\mathrm{Mpc}/h]$', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=7, loc='best', ncol=2)
    
    if grid:
        plt.grid(True, alpha=0.3)

    # Personnalisation of ticks of X axis if a mask is applied
    if mask_range is not None:
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # Ticks every 10 Mpc/h
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))  # Normal format (40, 50...)
    

    # If `yrange` is defined, fix Y axis limits
    if yrange is not None:
        plt.ylim(yrange)

    plt.show()

def plot_mass_comparison(
    r_hh_all_cs, quantity_all_cs, errors_cs,
    r_hh_all_rs, quantity_all_rs, errors_rs,
    mass_bins, bin_edges_exp, ylabel, title, 
    apply_transformation=None, mask_range=None,
    yrange = None, 
    rows=2, figsize_per_plot=(4, 5),
    scalex='log', scaley='log'
    ):

    n_bins = len(mass_bins) - 1  # Number of mass bins
    cols = (n_bins + rows - 1) // rows  # Compute the number of columns wrt the number of lines

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows), sharey=True)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(-0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=14, fontweight="bold")

    # Apply axes if necessary (because with rows/cols, axes becomes a 2D table)
    axes = axes.flatten()

    # Loop on mass bins
    for i in range(n_bins):
        ax = axes[i]

        # Define r and quantity for each bin 
        r_cs, r_rs = r_hh_all_cs[i], r_hh_all_rs[i]
        
        if errors_cs:
            quantity_cs, error_cs = quantity_all_cs[i]  # Get (values, errors)
        else:
            quantity_cs, error_cs = quantity_all_cs[i], None  # No errors

        if errors_rs:
            quantity_rs, error_rs = quantity_all_rs[i]  # Get (values, errors)
        else:
            quantity_rs, error_rs = quantity_all_rs[i], None  # No errors

        # Apply a transformation (ex: r^2 * ξ_0(r)) if necessary
        if apply_transformation is not None:
            quantity_cs = apply_transformation(r_cs, quantity_cs)
            quantity_rs = apply_transformation(r_rs, quantity_rs)

        # Appliqy a masque if necessary
        if mask_range is not None:
            mask_cs = (r_cs >= mask_range[0]) & (r_cs < mask_range[1])
            mask_rs = (r_rs >= mask_range[0]) & (r_rs < mask_range[1])
            r_cs = r_cs[mask_cs]
            r_rs = r_rs[mask_rs]
            quantity_cs = quantity_cs[mask_cs]
            quantity_rs = quantity_rs[mask_rs]
            if error_cs is not None:
                error_cs = error_cs[mask_cs]
            if error_rs is not None:
                error_rs = error_rs[mask_rs]

        # Plot with error bars if necessary
        if error_cs is not None:
            ax.errorbar(r_cs, quantity_cs, yerr=error_cs, fmt='o', color='green', label='CompaSo', capsize=3)
        else:
            ax.plot(r_cs, quantity_cs, linestyle='-', color='green', label='CompaSo')

        if error_rs is not None:
            ax.errorbar(r_rs, quantity_rs, yerr=error_rs, fmt='o', color='red', label='Rockstar', capsize=3)
        else:
            ax.plot(r_rs, quantity_rs, linestyle='-', color='red', label='Rockstar')

        # If `yrange` is defined, fix Y axis limits
        if yrange is not None:
            ax.set_ylim(yrange)

        # Specific title for each mass bin
        mass_min_exp, mass_max_exp = bin_edges_exp[i], bin_edges_exp[i + 1]
        ax.set_title(rf'Mass bin {i}: $10^{{{mass_min_exp:.1f}}}$ - $10^{{{mass_max_exp:.1f}}}$ $M_\odot$', fontsize=10)

        # Grid and legend
        if scalex =='log':
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

        if scaley =='log':
            ax.set_yscale('log')
            
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

        # Personnalisation of ticks of the X axis if a mask is applied
        # if mask_range is not None :
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  
        #     ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))  

    # Suppress empty subplots if 'n_bins' is not a mutliple of `rows * cols`
    for j in range(n_bins, len(axes)):
        fig.delaxes(axes[j])

    # Ajust labels
    for ax in axes[-cols:]:  # Dernière ligne uniquement
        ax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$', fontsize=14)
    
    # for row in range(rows):
    #     mid_col = cols // 2
    #     axes[row * cols + mid_col].set_ylabel(ylabel, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_ratio_comparison(
    r_hh_all_cs, quantity_all_cs, 
    r_hh_all_rs, quantity_all_rs, 
    mass_bins, bin_edges_exp, ylabel, title, 
    apply_transformation=None,
    mask_range=None, yrange = None,
    rows=2, figsize_per_plot=(4, 5)
):

    n_bins = len(mass_bins) - 1  # Number of mass_bins
    cols = (n_bins + rows - 1) // rows  # Compute the number of columns wrt the number of lines

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows), sharey=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axes = axes.flatten()  # Flatten the axis array for iteration

    for i in range(n_bins):
        ax = axes[i]

        # Retrieving r distances and values for CompaSo and Rockstar
        r_cs, r_rs = r_hh_all_cs[i], r_hh_all_rs[i]
        quantity_cs, quantity_rs = quantity_all_cs[i], quantity_all_rs[i]

        if apply_transformation is not None:
            quantity_cs = apply_transformation(r_cs, quantity_cs)
            quantity_rs = apply_transformation(r_rs, quantity_rs)
        
        # Apply a mask if provided
        if mask_range is not None:
            mask_cs = (r_cs >= mask_range[0]) & (r_cs < mask_range[1])
            mask_rs = (r_rs >= mask_range[0]) & (r_rs < mask_range[1])
            r_cs = r_cs[mask_cs]
            r_rs = r_rs[mask_rs]
            quantity_cs = quantity_cs[mask_cs]
            quantity_rs = quantity_rs[mask_rs]


        # Check that sizes match (adjust if necessary)
        # min_len = min(len(r_cs), len(r_rs))
        # r_cs, quantity_cs = r_cs[:min_len], quantity_cs[:min_len]
        # r_rs, quantity_rs = r_rs[:min_len], quantity_rs[:min_len]

        # Calculating the Rockstar / CompaSo ratio
        ratio = quantity_rs / quantity_cs
        ratio[np.isnan(ratio)] = np.nan  # Évite les NaN

        # Plot the ratio
        ax.plot(r_cs, ratio, linestyle='-', color='blue', label=r'Rockstar / CompaSo')

        # Add a specific title to the mass bin
        mass_min_exp, mass_max_exp = bin_edges_exp[i], bin_edges_exp[i + 1]
        ax.set_title(rf'Mass bin {i}: $10^{{{mass_min_exp:.1f}}}$ - $10^{{{mass_max_exp:.1f}}}$ $M_\odot$', fontsize=10)

        # If `yrange` is defined, fix Y axis limits
        if yrange is not None:
            ax.set_ylim(yrange)

        # Adding grids and reference lines
        ax.grid(True, alpha=0.3)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)  # Ligne y=1
        ax.legend(fontsize=8, loc='upper right')

        # Customise ticks X
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))  

    # Supprimer les subplots vides si `n_bins` < `rows * cols`
    for j in range(n_bins, len(axes)):
        fig.delaxes(axes[j])

    # Placer ylabel à gauche de tous les subplots
    fig.text(-0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=14, fontweight="bold")

    # Adjust X labels at the bottom only
    for ax in axes[-cols:]:  
        ax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$', fontsize=14)

    plt.tight_layout()
    plt.show()

def print_chosen_multipole_ratio(results_r, results_xi, ratio_indices, legends, ratio_legends, leg_loc, ratio_leg_loc, title, colors_list, ratio_colors, scalex=None, scaley=None, yrange=None, mask_range=None, ratio_yrange=None):

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'2PCF Comparison of {title}')

    fig.text(0.95, 0.5, 'Ratio', va='center', rotation='vertical', fontsize=14, color='blue')

    # Projection to monopole
    
    for r, xi, legend, color in zip(results_r, results_xi, legends, colors_list):
        ax.plot(r, r * r * xi.squeeze(), label=legend, linestyle="-", color = color)
    ax.legend(loc=leg_loc)
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$')
    if scalex =='log':
        ax.set_xscale('log')

    if scalex == 'log':
        ax.set_xscale('log')
    if scaley == 'log':
        ax.set_yscale('log')
    if yrange is not None:
        ax.set_ylim(yrange)
    if mask_range is not None:
        ax.set_xlim(mask_range)

    # Axe secondaire pour les ratios
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', labelcolor='blue')

    for (i, (i1, i2)) in enumerate(ratio_indices):
        legend = ratio_legends[i]
        color = ratio_colors[i]
        ratio_xi = np.where(results_xi[i2] != 0, results_xi[i1] / results_xi[i2], np.nan)
        ratio_xi[np.isnan(ratio_xi)] = np.nan
        ax2.plot(results_r[i2], ratio_xi.squeeze(), linestyle='--', label=legend, color=color)

    ax2.axhline(1, color="black", linestyle="--", linewidth=0.8)
    if ratio_yrange is not None:
        ax2.set_ylim(ratio_yrange)
    ax2.legend(fontsize=8, loc=ratio_leg_loc)


    plt.show()

def plot_mass_ratio_comparison(
    r_hh_all_cs, quantity_all_cs, errors_cs,
    r_hh_all_rs, quantity_all_rs, errors_rs,
    mass_bins, bin_edges_exp, ylabel, title, 
    ratio_leg, leg_pos,
    apply_transformation=None, mask_range=None,
    yrange = None, ratio_yrange=(0.8, 1.8),
    rows=2, figsize_per_plot=(4, 5),
    scalex='log', scaley='log'
    ):

    n_bins = len(mass_bins) - 1  # Number of mass bins
    cols = (n_bins + rows - 1) // rows  # Compute the number of columns wrt the number of lines

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows), sharey=True)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(-0.01, 0.5, ylabel, va='center', rotation='vertical', fontsize=14, fontweight="bold")
    fig.text(1, 0.5, 'Ratio', va='center', rotation='vertical', fontsize=14, color='blue')

    # Apply axes if necessary (because with rows/cols, axes becomes a 2D table)
    axes = axes.flatten()

    # Loop on mass bins
    for i in range(n_bins):
        ax = axes[i]

        # Define r and quantity for each bin 
        r_cs, r_rs = r_hh_all_cs[i], r_hh_all_rs[i]
        
        if errors_cs:
            quantity_cs, error_cs = quantity_all_cs[i]  # Get (values, errors)
        else:
            quantity_cs, error_cs = quantity_all_cs[i], None  # No errors

        if errors_rs:
            quantity_rs, error_rs = quantity_all_rs[i]  # Get (values, errors)
        else:
            quantity_rs, error_rs = quantity_all_rs[i], None  # No errors

        # Apply a transformation (ex: r^2 * ξ_0(r)) if necessary
        if apply_transformation is not None:
            quantity_cs = apply_transformation(r_cs, quantity_cs)
            quantity_rs = apply_transformation(r_rs, quantity_rs)

        # Appliqy a masque if necessary
        if mask_range is not None:
            mask_cs = (r_cs >= mask_range[0]) & (r_cs < mask_range[1])
            mask_rs = (r_rs >= mask_range[0]) & (r_rs < mask_range[1])
            r_cs = r_cs[mask_cs]
            r_rs = r_rs[mask_rs]
            quantity_cs = quantity_cs[mask_cs]
            quantity_rs = quantity_rs[mask_rs]
            if error_cs is not None:
                error_cs = error_cs[mask_cs]
            if error_rs is not None:
                error_rs = error_rs[mask_rs]

        # Plot with error bars if necessary
        if error_cs is not None:
            ax.errorbar(r_cs, quantity_cs, yerr=error_cs, fmt='o', color='green', label='CompaSo', capsize=3)
        else:
            ax.plot(r_cs, quantity_cs, linestyle='-', color='green', label='CompaSo')

        if error_rs is not None:
            ax.errorbar(r_rs, quantity_rs, yerr=error_rs, fmt='o', color='red', label='Rockstar', capsize=3)
        else:
            ax.plot(r_rs, quantity_rs, linestyle='-', color='red', label='Rockstar')

        # If `yrange` is defined, fix Y axis limits
        if yrange is not None:
            ax.set_ylim(yrange)

        # Specific title for each mass bin
        mass_min_exp, mass_max_exp = bin_edges_exp[i], bin_edges_exp[i + 1]
        ax.set_title(rf'Mass bin {i}: $10^{{{mass_min_exp:.1f}}}$ - $10^{{{mass_max_exp:.1f}}}$ $M_\odot$', fontsize=10)

        # Grid and legend
        if scalex =='log':
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())

        if scaley =='log':
            ax.set_yscale('log')
            
        ax.grid(True, alpha=0.3)

        # Personnalisation of ticks of the X axis if a mask is applied
        # if mask_range is not None :
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  
        #     ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))  

        # Twin axis for ratio
        ax2 = ax.twinx()
        if len(quantity_cs) == len(quantity_rs):
            ratio = quantity_rs / quantity_cs
            ratio[np.isnan(ratio)] = np.nan
            ax2.plot(r_cs, ratio, color='blue', linestyle='--', label='Ratio RS/CS')
            plt.axhline(1, color="black", linestyle="--", linewidth=0.5)
            # ax2.set_ylabel('Ratio', fontsize=10, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_ylim(ratio_yrange)
            ax2.legend(fontsize=8, loc=ratio_leg)

        ax.legend(fontsize=8, loc=leg_pos)

    # Suppress empty subplots if 'n_bins' is not a mutliple of `rows * cols`
    for j in range(n_bins, len(axes)):
        fig.delaxes(axes[j])

    # Ajust labels
    for ax in axes[-cols:]:  # Dernière ligne uniquement
        ax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$', fontsize=14)
    
    # for row in range(rows):
    #     mid_col = cols // 2
    #     axes[row * cols + mid_col].set_ylabel(ylabel, fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_multipole_ratio(r, xi_cs, xi_rs, title, l, label, yrange, mask_range, margin_factor=0.05):
    """
    Plot the ratio of xi_rs to xi_cs instead of the difference, centered around y=1,
    with added margin.
    """
    # Compute the ratio, avoiding division by zero
    ratio_xi = np.where(xi_cs != 0, xi_rs / xi_cs, np.nan)

    # Déterminer les nouvelles limites de l'axe Y centrées sur 1
    y_min, y_max = np.nanmin(ratio_xi), np.nanmax(ratio_xi)  # Ignorer les NaN
    max_dev = max(abs(y_max - 1), abs(1 - y_min))  # Distance max par rapport à 1

    # Ajouter une marge pour éviter que les points touchent les bords
    margin = margin_factor * max_dev  
    new_y_min = 1 - max_dev - margin
    new_y_max = 1 + max_dev + margin

    # Tracé
    plt.figure(figsize=(7, 5))
    plt.plot(r, ratio_xi.squeeze(), linestyle="-", color="blue", label=label)
    plt.axhline(1, color="black", linestyle="--", linewidth=1)  # Ligne y=1 pour référence

    if yrange is not None:
        plt.ylim(yrange)
    # plt.ylim(new_y_min, new_y_max)  # Ajuster l'axe Y autour de 1 avec marge
    if mask_range is not None:
        plt.xlim(mask_range)

    plt.xlabel(r"$s$ [Mpc/h]", fontsize=14)
    plt.ylabel(r"$\xi_{\ell=" + str(l) + r"}^{RS} / \xi_{\ell=" + str(l) + r"}^{CS}$", fontsize=14)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.show()

## BIAS 

def bias_wrt_mass(mass_bins, sampled_masses, positions, bins_s, bins_mu, boxsize, xi_mm_mono, xi_mm_quad, save_path, mask_range=(40, 80), nthreads=32):

    bias_by_mass_mono = []
    bias_by_mass_quad = []

    # Store the results of r and xi for each bin
    r_hh_all = []
    xi_hh_mono_all = []
    xi_hh_quad_all = []
    bias_by_r_mono_all = []  # complete list of biases by r
    bias_by_r_quad_all = []  

    for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):

        # Selecting halos in this mass bin
        mask = (sampled_masses >= low) & (sampled_masses < high)
        halos_in_bin = positions[mask]

        # Check for halos in the bin
        if halos_in_bin.shape[0] == 0:
            print(f"Skipping bin {i} because it contains no halos.")
            continue

        print(f"Bin {i}: {halos_in_bin.shape[0]} halos")

        # Separate x, y, z if necessary
        if halos_in_bin.shape[1] == 3:
            halos_in_bin = [halos_in_bin[:, 0], halos_in_bin[:, 1], halos_in_bin[:, 2]]

        # Calculation of the halo-halo correlation function
        results_hh = TwoPointCorrelationFunction(
            mode='smu',
            edges=(bins_s, bins_mu),
            data_positions1=halos_in_bin,
            boxsize=boxsize,
            nthreads=nthreads,
            los='z'
        )
        r_hh, xi_hh_mono = results_hh(ells=(0,), return_sep=True)
        _, xi_hh_quad = results_hh(ells=(2,), return_sep=True)

        xi_hh_mono_sq = xi_hh_mono.squeeze()
        xi_hh_quad_sq = xi_hh_quad.squeeze()
        xi_mm_mono_sq = xi_mm_mono.squeeze()
        xi_mm_quad_sq = xi_mm_quad.squeeze()

        if mask_range == (None, None):
            mask = np.ones_like(r_hh, dtype=bool)  
        else:
            mask = (r_hh >= mask_range[0]) & (r_hh < mask_range[1])

        # Calculation of bias
        bias_mono = xi_hh_mono_sq[mask] / xi_mm_mono_sq[mask]
        bias_quad = xi_hh_quad_sq[mask] / xi_mm_quad_sq[mask]

        bias_by_mass_mono.append((np.mean([low, high]), np.mean(bias_mono)))
        bias_by_mass_quad.append((np.mean([low, high]), np.mean(bias_quad)))

        # Store results for this bin
        r_hh_all.append(r_hh[mask])
        xi_hh_mono_all.append(xi_hh_mono_sq[mask])
        xi_hh_quad_all.append(xi_hh_quad_sq[mask])
        bias_by_r_mono_all.append(bias_mono)  # Adding bias as a function of r
        bias_by_r_quad_all.append(bias_quad)

    # Saving results
    np.savez(save_path, 
             mass_bins=mass_bins, 
             r_hh_all=r_hh_all, 
             xi_hh_mono_all=xi_hh_mono_all, 
             xi_hh_quad_all=xi_hh_quad_all,
             bias_by_mass_mono=bias_by_mass_mono,  # Moyenne du biais par bin
             bias_by_mass_quad=bias_by_mass_quad,
             bias_by_r_mono_all=bias_by_r_mono_all,  # Nouveau : biais par r
             bias_by_r_quad_all=bias_by_r_quad_all)
    
    print(f"\n Fichier enregistré : {save_path}")

    return bias_by_mass_mono, bias_by_mass_quad

def bootstrap_bias_wrt_mass(
        mass_bins, sampled_masses, positions, bins_s, bins_mu, boxsize, 
        xi_mm_mono, xi_mm_quad, mask_range=(40, 80), n_bootstrap=100, nthreads=32, save_path=None):
    
    # Store results
    r_hh_all = []
    xi_hh_mono_all = []
    xi_hh_quad_all = []
    
    bias_by_r_mono_all = []  # List of biases as a function of r for each bootstrap
    bias_by_r_quad_all = []
    
    mean_bias_mono_all = []  # Bootstrap mean and standard deviation by mass bin
    mean_bias_quad_all = []

    # Loop on ground bins
    for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask = (sampled_masses >= low) & (sampled_masses < high)
        halos_in_bin = positions[mask]

        if halos_in_bin.shape[0] == 0:
            print(f"Skipping mass bin {i} ({low:.2e} - {high:.2e}) : No halos found.")
            continue

        print(f"Mass bin {i}: {halos_in_bin.shape[0]} halos")

        # Separate x, y, z
        halos_in_bin = halos_in_bin.T  # Transforms into (3, N) format
        
        # Initialise lists to store bootstrap biases
        bootstrapped_bias_r_mono = []
        bootstrapped_bias_r_quad = []

        # Bootstrap
        for _ in range(n_bootstrap):
            resampled_indices = np.random.randint(0, halos_in_bin.shape[1], halos_in_bin.shape[1])
            resampled_halos = halos_in_bin[:, resampled_indices]  # (3, N)

            # Calculation of the halo-halo correlation function
            results_hh = TwoPointCorrelationFunction(
                mode='smu',
                edges=(bins_s, bins_mu),
                data_positions1=[resampled_halos[0], resampled_halos[1], resampled_halos[2]],
                boxsize=boxsize,
                nthreads=nthreads,
                los='z'
            )
            r_hh, xi_hh_mono = results_hh(ells=(0,), return_sep=True)
            _, xi_hh_quad = results_hh(ells=(2,), return_sep=True)

            # Apply the mask by r
            if mask_range == (None, None):
                mask_r = np.ones_like(r_hh, dtype=bool)  
            else:
                mask_r = (r_hh >= mask_range[0]) & (r_hh < mask_range[1])

            # Calculation of bias by r
            bias_r_mono = xi_hh_mono.squeeze()[mask_r] / xi_mm_mono.squeeze()[mask_r]
            bias_r_quad = xi_hh_quad.squeeze()[mask_r] / xi_mm_quad.squeeze()[mask_r]

            # Store bias by r for this bootstrap run
            bootstrapped_bias_r_mono.append(bias_r_mono)
            bootstrapped_bias_r_quad.append(bias_r_quad)

        # Convert to numpy array
        bootstrapped_bias_r_mono = np.array(bootstrapped_bias_r_mono)  # (n_bootstrap, n_r)
        bootstrapped_bias_r_quad = np.array(bootstrapped_bias_r_quad)

        # Store bootstrap biases for each bin
        bias_by_r_mono_all.append(bootstrapped_bias_r_mono)
        bias_by_r_quad_all.append(bootstrapped_bias_r_quad)

        # Mean and standard deviation of bootstrap runs
        mean_bias_mono_all.append((np.mean(bootstrapped_bias_r_mono, axis=0), np.std(bootstrapped_bias_r_mono, axis=0)))
        mean_bias_quad_all.append((np.mean(bootstrapped_bias_r_quad, axis=0), np.std(bootstrapped_bias_r_quad, axis=0)))

        # Store r and xi
        r_hh_all.append(r_hh[mask_r])
        xi_hh_mono_all.append(xi_hh_mono.squeeze()[mask_r])
        xi_hh_quad_all.append(xi_hh_quad.squeeze()[mask_r])

    # Saving results
    if save_path:
        np.savez(save_path, 
                 mass_bins=mass_bins, 
                 r_hh_all=r_hh_all, 
                 xi_hh_mono_all=xi_hh_mono_all, 
                 xi_hh_quad_all=xi_hh_quad_all, 
                 bias_by_r_mono_all=bias_by_r_mono_all,  
                 bias_by_r_quad_all=bias_by_r_quad_all,  
                 mean_bias_mono_all=mean_bias_mono_all,  
                 mean_bias_quad_all=mean_bias_quad_all)
        
        print(f"\n Fichier enregistré : {save_path}")

    return mean_bias_mono_all, mean_bias_quad_all

def removing_bias_wrt_mass(
    mass_bins, sampled_masses, positions, bins_s, bins_mu, boxsize, 
    xi_mm_mono, xi_mm_quad, mask_range=(40, 80), n_bootstrap=100, nthreads=32, save_path=None):

    # Stocker les résultats
    r_hh_all = []
    xi_hh_mono_all = []
    xi_hh_quad_all = []

    bias_by_r_mono_all = []  
    bias_by_r_quad_all = []

    mean_bias_mono_all = []  
    mean_bias_quad_all = []

    for i, (low, high) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        mask = (sampled_masses >= low) & (sampled_masses < high)
        halos_in_bin = positions[mask]

        if halos_in_bin.shape[0] == 0:
            print(f"Skipping mass bin {i} ({low:.2e} - {high:.2e}) : No halos found.")
            continue

        print(f"Mass bin {i}: {halos_in_bin.shape[0]} halos")

        halos_in_bin = halos_in_bin.T  # format (3, N)
        
        results_hh_full = TwoPointCorrelationFunction(
            mode='smu',
            edges=(bins_s, bins_mu),
            data_positions1=[halos_in_bin[0], halos_in_bin[1], halos_in_bin[2]],
            boxsize=boxsize,
            nthreads=nthreads,
            los='z'
        )

        r_hh_full, xi_hh_mono_full = results_hh_full(ells=(0,), return_sep=True)
        _, xi_hh_quad_full = results_hh_full(ells=(2,), return_sep=True)

        # Apply the mask by r
        if mask_range == (None, None):
            mask_r = np.ones_like(r_hh_full, dtype=bool)  
        else:
            mask_r = (r_hh_full >= mask_range[0]) & (r_hh_full < mask_range[1])

        # Initialiser les listes pour stocker les biais après suppression de 10%
        bias_mono_removed = []
        bias_quad_removed = []

        # 10% of halos removed at random
        for _ in range(n_bootstrap):
            num_remove = int(0.1 * halos_in_bin.shape[1])  # 10% of halos
            indices_keep = np.random.choice(halos_in_bin.shape[1], size=halos_in_bin.shape[1] - num_remove, replace=False)
            resampled_halos = halos_in_bin[:, indices_keep]  # (3, N')

            # Calculation of 2PCF with 10% less
            results_hh_removed = TwoPointCorrelationFunction(
                mode='smu',
                edges=(bins_s, bins_mu),
                data_positions1=[resampled_halos[0], resampled_halos[1], resampled_halos[2]],
                boxsize=boxsize,
                nthreads=nthreads,
                los='z'
            )
            r_hh_removed, xi_hh_mono_removed = results_hh_removed(ells=(0,), return_sep=True)
            _, xi_hh_quad_removed = results_hh_removed(ells=(2,), return_sep=True)

            # Appliquer le masque en r
            # Apply the mask by r
            if mask_range == (None, None):
                mask_r_removed = np.ones_like(r_hh_removed, dtype=bool)  
            else:
                mask_r_removed = (r_hh_removed >= mask_range[0]) & (r_hh_removed < mask_range[1])

            # Calculation of bias after suppression
            bias_r_mono = xi_hh_mono_removed.squeeze()[mask_r_removed] / xi_mm_mono.squeeze()[mask_r_removed]
            bias_r_quad = xi_hh_quad_removed.squeeze()[mask_r_removed] / xi_mm_quad.squeeze()[mask_r_removed]

            bias_mono_removed.append(bias_r_mono)
            bias_quad_removed.append(bias_r_quad)

        bias_mono_removed = np.array(bias_mono_removed)  # (n_bootstrap, n_r)
        bias_quad_removed = np.array(bias_quad_removed)

        bias_by_r_mono_all.append(bias_mono_removed)
        bias_by_r_quad_all.append(bias_quad_removed)

        mean_bias_mono_all.append((np.mean(bias_mono_removed, axis=0), np.std(bias_mono_removed, axis=0)))
        mean_bias_quad_all.append((np.mean(bias_quad_removed, axis=0), np.std(bias_quad_removed, axis=0)))

        r_hh_all.append(r_hh_full[mask_r])
        xi_hh_mono_all.append(xi_hh_mono_full.squeeze()[mask_r])
        xi_hh_quad_all.append(xi_hh_quad_full.squeeze()[mask_r])

    # Sauvegarde des résultats
    if save_path:
        np.savez(save_path, 
                    mass_bins=mass_bins, 
                    r_hh_all=r_hh_all, 
                    xi_hh_mono_all=xi_hh_mono_all, 
                    xi_hh_quad_all=xi_hh_quad_all, 
                    bias_by_r_mono_all=bias_by_r_mono_all,  
                    bias_by_r_quad_all=bias_by_r_quad_all,  
                    mean_bias_mono_all=mean_bias_mono_all,  
                    mean_bias_quad_all=mean_bias_quad_all)
        
        print(f"\n Fichier enregistré : {save_path}")

    return mean_bias_mono_all, mean_bias_quad_all

## MASS DENSITY

def sigma_integral(chi, r, xi_gm_interp):
    return xi_gm_interp(np.sqrt(r ** 2 + chi ** 2))

def sigma_mean(r_prime, xi_gm_interp):
    return xi_gm_interp(r_prime) * (r_prime)

def compute_HOD_lensing(r_gm_cs, xi_gm, rho_crit, limit=40):

    r_gm_cs = np.where(r_gm_cs == 0, 1e-10, r_gm_cs)

    xi_gm = xi_gm.flatten()
    last_value = float(xi_gm[-1])

    # Création du spline pour xi_gm
    xi_gm_interp = interp1d(r_gm_cs, xi_gm, kind='cubic', bounds_error=False, fill_value=(last_value,0)) # fill_value=(last_value,0)
        
    # Calcul de SIGMA et SIGMA_MEAN
    SIGMA = 2 * np.array([quad(sigma_integral, 1e-4, 100, args=(x, xi_gm_interp), limit=limit, full_output=1)[0] for x in r_gm_cs]) * rho_crit
    
    spline_sigma = interp1d(r_gm_cs, SIGMA, kind='cubic', bounds_error=False, fill_value=(SIGMA[0], 0))
    SIGMA_MEAN = np.array([quad(sigma_mean, 1e-4, x, args=(spline_sigma),limit=limit, full_output=1)[0] for x in r_gm_cs]) * 2 / r_gm_cs ** 2

    # Calcul de Delta_Sigma
    Delta_Sigma = (SIGMA_MEAN - SIGMA)

    return Delta_Sigma, SIGMA_MEAN, SIGMA