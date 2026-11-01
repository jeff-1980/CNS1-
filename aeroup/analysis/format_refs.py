"""Convert author-year cites in manuscript_v1.md to Elsevier numbered cites and append the reference list.
Usage: python format_refs.py <in.md> <out.md>"""
import re
import sys

REFS = {
 "zhao": "Z. Zhao, R. Jiang, J. Feng, H. Liu, T. Wang, W. Shen, M. Chen, D. Wang, Y. Liu, Researches on vortex generators applied to wind turbines: a review, Ocean Eng. 253 (2022) 111266. https://doi.org/10.1016/j.oceaneng.2022.111266.",
 "mueller": "H. Mueller-Vahl, G. Pechlivanoglou, C.N. Nayeri, C.O. Paschereit, Vortex generators for wind turbine blades: a combined wind tunnel and wind turbine parametric study, in: Proc. ASME Turbo Expo 2012: Turbine Technical Conference and Exposition, Vol. 6, Copenhagen, 2012, pp. 899–914. https://doi.org/10.1115/GT2012-69197.",
 "astolfi18": "D. Astolfi, F. Castellani, L. Terzi, Wind turbine power curve upgrades, Energies 11 (2018) 1300. https://doi.org/10.3390/en11051300.",
 "astolfi19": "D. Astolfi, F. Castellani, M.L. Fravolini, S. Cascianelli, L. Terzi, Precision computation of wind turbine power upgrades: an aerodynamic and control optimization test case, J. Energy Resour. Technol. 141 (2019) 051205. https://doi.org/10.1115/1.4042450.",
 "terzi": "L. Terzi, A. Lombardi, F. Castellani, D. Astolfi, Innovative methods for wind turbine power curve upgrade assessment, J. Phys. Conf. Ser. 1102 (2018) 012036. https://doi.org/10.1088/1742-6596/1102/1/012036.",
 "castellani": "F. Castellani, D. Astolfi, L. Terzi, Wind turbine power curve upgrades: methods for the assessment and test cases study, J. Phys. Conf. Ser. 1452 (2020) 012004. https://doi.org/10.1088/1742-6596/1452/1/012004.",
 "manolesos": "M. Manolesos, Y. Celik, H. Ramsay, R. Karande, B. Wood, I. Dinwoodie, I. Masters, M. Harrold, G. Papadakis, Performance improvement of a Vestas V52 850kW wind turbine by retrofitting passive flow control devices, J. Phys. Conf. Ser. 2767 (2024) 022027. https://doi.org/10.1088/1742-6596/2767/2/022027.",
 "moon": "H. Moon, J. Jeong, S. Park, K. Ha, J.-H. Jeong, Numerical and experimental validation of vortex generator effect on power performance improvement in MW-class wind turbine blade, Renew. Energy 212 (2023) 443–454. https://doi.org/10.1016/j.renene.2023.04.104.",
 "iec": "IEC 61400-12-1:2022, Wind energy generation systems – Part 12-1: Power performance measurements of electricity producing wind turbines, Edition 3.0, International Electrotechnical Commission, Geneva, 2022.",
 "clerc": "A. Clerc, A. Torner, A. Bamberger, M. Lannic, E. Lingkan, S. Northover-Naylor, G. Powles, K. Watts, T. Young, Open-source dataset and AEP validation from a real-world case study of turbine upgrades worth 5% AEP, presentation at WindEurope Technology Workshop 2025, Istanbul, 20 June 2025. https://www.res-group.com/wp-content/uploads/2026/05/RES-open-source-AEP-validation-of-turbine-upgrades.pdf (accessed 1 November 2026).",
 "ribrant": "J. Ribrant, L.M. Bertling, Survey of failures in wind power systems with focus on Swedish wind power plants during 1997–2005, IEEE Trans. Energy Convers. 22 (2007) 167–173. https://doi.org/10.1109/TEC.2006.889614.",
 "spinato": "F. Spinato, P.J. Tavner, G.J.W. van Bussel, E. Koutoulakos, Reliability of wind turbine subassemblies, IET Renew. Power Gener. 3 (2009) 387–401. https://doi.org/10.1049/iet-rpg.2008.0060.",
 "pinar": "J.M. Pinar Pérez, F.P. García Márquez, A. Tobias, M. Papaelias, Wind turbine reliability analysis, Renew. Sustain. Energy Rev. 23 (2013) 463–472. https://doi.org/10.1016/j.rser.2013.03.018.",
 "artigao": "E. Artigao, S. Martín-Martínez, A. Honrubia-Escribano, E. Gómez-Lázaro, Wind turbine reliability: a comprehensive review towards effective condition monitoring development, Appl. Energy 228 (2018) 1569–1583. https://doi.org/10.1016/j.apenergy.2018.07.037.",
 "kim": "S. Kim, H. Im, B. Kim, Numerical analysis of the effects of vortex generators attached to contaminated blades on wind turbine power performance and fatigue load, J. Mech. Sci. Technol. 37 (2023) 3515–3524. https://doi.org/10.1007/s12206-023-0617-0.",
 "bak": "C. Bak, W. Skrzypiński, M. Gaunaa, H. Villanueva, N.F. Brønnum, E.K. Kruse, Full scale wind turbine test of vortex generators mounted on the entire blade, J. Phys. Conf. Ser. 753 (2016) 022001. https://doi.org/10.1088/1742-6596/753/2/022001.",
 "ji": "F. Ji, F. Yang, Z. Wang, S. Liu, D. Xu, W. Zhou, L. Wu, X. Chu, Y. Zhong, Y. Ke, Modular performance testing and comparative evaluation method for wind turbine retrofit schemes, Machines 14 (2026) 965. https://doi.org/10.3390/machines14090965.",
 "hot": "A. Clerc, E. Lingkan, Hill of Towie wind farm open dataset, version 2.1.0 [dataset], Zenodo, 2026. https://doi.org/10.5281/zenodo.22662930.",
 "iso281": "ISO 281:2007, Rolling bearings – Dynamic load ratings and rating life, International Organization for Standardization, Geneva, 2007.",
 "wang": "Q. Wang, Y. Vidal, Neighbor-based estimation of wind turbine power for upgrade performance assessment, J. Phys. Conf. Ser. 3232 (2026) 012001. https://doi.org/10.1088/1742-6596/3232/1/012001.",
}
KEYMAP = [("zhao", "zhao"), ("mueller", "mueller"), ("astolfi 2018 preprint", None), ("astolfi 2019", "astolfi19"),
          ("astolfi 2018", "astolfi18"), ("terzi", "terzi"), ("castellani", "castellani"), ("manolesos", "manolesos"),
          ("moon", "moon"), ("iec", "iec"), ("clerc", "clerc"), ("ribrant", "ribrant"), ("spinato", "spinato"),
          ("pinar", "pinar"), ("artigao", "artigao"), ("kim", "kim"), ("bak", "bak"), ("ji 2026", "ji"),
          ("hot dataset", "hot"), ("iso 281", "iso281"), ("wang 2026", "wang")]


def key(tok):
    t = tok.strip().lower()
    for k, v in KEYMAP:
        if t.startswith(k):
            return v
    raise KeyError(tok)


src = open(sys.argv[1], encoding="utf-8").read()
body, _ = src.split("## References", 1)
order = []
CITE = re.compile(r"\[((?:[A-Z][^\[\]]*?(?:19|20)\d\d[^\[\]]*?)|IEC 61400-12-1|ISO 281|HoT dataset[^\]]*)\]")


def sub(m):
    ks = []
    for tok in m.group(1).split(";"):
        k = key(tok)
        if k and k not in ks:
            ks.append(k)
    for k in ks:
        if k not in order:
            order.append(k)
    nums = sorted(order.index(k) + 1 for k in ks)
    # Elsevier: [1,2] or [3–5] for runs
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i:j + 1])))
        i = j + 1
    return "[" + ",".join(parts) + "]"


body = CITE.sub(sub, body)
missing = set(REFS) - set(order)
reflist = "## References\n\n" + "\n".join(f"[{i + 1}] {REFS[k]}\n" for i, k in enumerate(order))
open(sys.argv[2], "w", encoding="utf-8").write(body + reflist)
print("cited", len(order), "uncited:", missing)
