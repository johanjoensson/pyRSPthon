from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pyRSPthon.cli._common import add_plot_arguments, apply_plot_style, finish_plots
from pyRSPthon.read import extract_dos


def run(args):
    import matplotlib.pyplot as plt

    dos = extract_dos(prefix=args.directory)
    _ = plt.figure()
    plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
    if dos.up is not None:
        plt.plot(dos.w, dos.up, label=r"$\uparrow$")
        plt.plot(dos.w, dos.down, "--", label=r"$\downarrow$")
    plt.title("Density of states")
    plt.xlabel(r"E - E$_F$")
    plt.ylabel(r"DOS")
    plt.legend()
    for name, data in (("S", dos.s), ("L", dos.l), ("J", dos.j)):
        if data is None:
            continue
        _ = plt.figure()
        plt.fill_between(dos.w, dos.sum, alpha=0.2, color="tab:gray", label="Total")
        plt.plot(dos.w, data[:, 2], label=rf"{name}$_z$")
        plt.title("Density of states")
        plt.xlabel(r"E - E$_F$")
        plt.ylabel(r"DOS")
        plt.legend()
    finish_plots(args)


def main():
    parser = ArgumentParser(
        description="Generate a Density of States (DOS) plot from RSPt output (dos.dat).\n\n"
                    "Physics Conventions:\n"
                    "- Energy reference: E - E_F = 0 (Fermi energy is shifted to 0).\n"
                    "- Spin: Up (solid) is positive, Down (dashed) is plotted on the same positive scale.\n"
                    "- Smearing: Uses the numerical smearing already applied by RSPt in dos.dat.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_plot_arguments(parser)
    args = parser.parse_args()
    apply_plot_style(args)
    run(args)


if __name__ == "__main__":
    main()
