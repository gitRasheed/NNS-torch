import torch

from nns_torch import experimental as exp


def main():
    x = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    y = torch.tensor([2.0, -3.0, 4.0, -1.0])
    comps = exp.asym_pm_components(x, y, target_x=0.0, target_y=0.0, degree=2.0, dim=0)

    symmetric = exp.asym_pm_cor(x, y, 0.0, 0.0, degree=2.0, weights=(1.0, 1.0, -1.0, -1.0), dim=0)
    ul_sensitive = exp.asym_pm_cor(x, y, 0.0, 0.0, degree=2.0, weights=(1.0, 1.0, -2.0, 0.0), dim=0)

    print(f"UU={comps.uu.item():.3f} LL={comps.ll.item():.3f} UL={comps.ul.item():.3f} LU={comps.lu.item():.3f}")
    print(f"symmetric_score={symmetric.item():.3f}")
    print(f"ul_sensitive_score={ul_sensitive.item():.3f}")


if __name__ == "__main__":
    main()
