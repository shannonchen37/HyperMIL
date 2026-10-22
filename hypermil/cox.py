from __future__ import annotations

import torch


def cox_efron_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative partial log likelihood over the complete risk set with Efron ties."""
    risk = risk.reshape(-1)
    time = time.reshape(-1).to(risk.device)
    event = event.reshape(-1).to(risk.device)
    event_times = torch.unique(time[event > 0])
    log_likelihood = risk.new_zeros(())
    event_count = event.sum()
    if int(event_count.item()) == 0:
        raise ValueError("Cox loss requires at least one observed event")
    for current in event_times:
        tied = (time == current) & (event > 0)
        d = int(tied.sum().item())
        risk_set = time >= current
        tied_exp = torch.exp(risk[tied]).sum()
        risk_exp = torch.exp(risk[risk_set]).sum()
        log_likelihood = log_likelihood + risk[tied].sum()
        for offset in range(d):
            log_likelihood = log_likelihood - torch.log(risk_exp - (offset / d) * tied_exp)
    return -log_likelihood / event_count

