"""A categorical evidence field with recurrent learned set routing.

The inputs are item identities and *context* rating categories, not positive or
negative labels. Learned ports pool observed context states, then send messages
to every candidate item. Finite convex updates keep states bounded; this is not
a physical fluid, a convergent solver, or an identified model of trust.

This implementation is independent of the project's fitted recommenders. Its
intermediate attention vectors belong to the established set-attention family.
"""
import math

import torch
from torch import nn


VARIANTS = ('adaptive', 'fixed_flow', 'hard_clamp', 'fixed_routing', 'fixed_fidelity')


class CategoricalEvidenceField(nn.Module):
    """Predict five rating categories from a partially observed rating context.

    ``context_ratings`` has shape [users, items], integer values 0..5. Zero is
    absent evidence, and item zero is padding. Remove every probe value *and*
    observation indicator before calling this method. The model never receives
    probe labels, a user identifier, metadata, or a fitted expert's predictions.

    ``fixed_flow`` freezes initial routing and source gates within each forward
    pass, while ``hard_clamp`` restores each observed state after each update.
    All variants allocate the same parameters. Their effective active capacity
    differs: the source-gate parameters have no effect under hard clamping.
    """

    def __init__(self, n_items, dim=8, ports=16, steps=4,
                 variant='adaptive', damping=.5):
        super().__init__()
        if any(type(value) is not int or value < 1 for value in (n_items, dim, ports, steps)):
            raise ValueError('Model dimensions and steps must be positive integers')
        if n_items < 2 or variant not in VARIANTS or not 0 < damping <= 1:
            raise ValueError('Invalid catalog, variant, or convex damping')
        self.config = dict(n_items=n_items, dim=dim, ports=ports, steps=steps,
                           variant=variant, damping=float(damping))
        self.n_items, self.dim, self.steps = n_items, dim, steps
        self.variant, self.damping = variant, float(damping)
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.rating_embedding = nn.Embedding(6, dim, padding_idx=0)
        self.ports = nn.Parameter(torch.empty(ports, dim))
        self.source_interaction = nn.Linear(dim, dim, bias=False)
        self.item_route = nn.Linear(dim, dim, bias=False)
        self.state_route = nn.Linear(dim, dim, bias=False)
        self.port_value = nn.Linear(dim, dim, bias=False)
        self.source_gate = nn.Sequential(nn.Linear(4 * dim, dim), nn.Tanh(),
                                         nn.Linear(dim, 1))
        self.decoder = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.Tanh(),
                                     nn.Linear(2 * dim, 5))
        nn.init.normal_(self.item_embedding.weight, std=.2)
        nn.init.normal_(self.rating_embedding.weight, std=.2)
        nn.init.normal_(self.ports, std=1.)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.rating_embedding.weight[0].zero_()

    def _routing(self, item_keys, state, observed):
        logits = (item_keys + self.state_route(state)) @ self.ports.T / math.sqrt(self.dim)
        down = logits.softmax(dim=-1)
        # The upward normalization has support only on observed context items.
        # An empty context receives zero weights instead of an all-masked NaN.
        up_logits = logits.transpose(1, 2).masked_fill(~observed[:, None, :], -torch.inf)
        any_observed = observed.any(dim=1)[:, None, None]
        safe_logits = torch.where(any_observed, up_logits, torch.zeros_like(up_logits))
        up = safe_logits.softmax(dim=-1) * observed[:, None, :]
        return up, down

    def _message(self, state, up, down):
        port_states = torch.bmm(up, state)
        return torch.bmm(down, torch.tanh(self.port_value(port_states)))

    def _gate(self, items, rating_values, state, message, observed):
        features = torch.cat((items, rating_values, state, message), dim=-1)
        return torch.sigmoid(self.source_gate(features)) * observed.unsqueeze(-1)

    def _decode(self, items, state):
        return self.decoder(torch.cat((items, state, items * state), dim=-1))

    def forward(self, context_ratings, return_diagnostics=False):
        if (context_ratings.dtype != torch.long or context_ratings.ndim != 2
                or context_ratings.shape[1] != self.n_items or context_ratings.shape[0] == 0):
            raise ValueError('Require a nonempty LongTensor with shape [users, n_items]')
        if torch.any((context_ratings < 0) | (context_ratings > 5)):
            raise ValueError('Context categories must be in 0..5')
        if torch.any(context_ratings[:, 0] != 0):
            raise ValueError('Padding item zero cannot contain evidence')
        observed = context_ratings != 0
        items = self.item_embedding.weight.unsqueeze(0).expand(context_ratings.shape[0], -1, -1)
        ratings = self.rating_embedding(context_ratings)
        source = torch.tanh(ratings + self.source_interaction(items * ratings))
        source = source * observed.unsqueeze(-1)
        state = source
        item_keys = self.item_route(items)
        cached_up = cached_down = cached_gate = None
        histories = {'states': [state], 'up': [], 'down': [], 'fidelity': [],
                     'rating_probabilities': [self._decode(items, state).softmax(-1)]} if return_diagnostics else None
        item_mask = torch.ones_like(observed)
        item_mask[:, 0] = False
        for step in range(self.steps):
            if step == 0 or self.variant not in ('fixed_flow', 'fixed_routing'):
                up, down = self._routing(item_keys, state, observed)
                if step == 0:
                    cached_up, cached_down = up, down
            else:
                up, down = cached_up, cached_down
            message = self._message(state, up, down)
            if step == 0 or self.variant not in ('fixed_flow', 'fixed_fidelity'):
                gate = self._gate(items, ratings, state, message, observed)
                if step == 0:
                    cached_gate = gate
            else:
                gate = cached_gate
            target = (1 - gate) * message + gate * source
            state = (1 - self.damping) * state + self.damping * target
            if self.variant == 'hard_clamp':
                state = torch.where(observed.unsqueeze(-1), source, state)
            state = state * item_mask.unsqueeze(-1)
            if return_diagnostics:
                histories['states'].append(state)
                histories['up'].append(up)
                histories['down'].append(down)
                histories['fidelity'].append(gate.squeeze(-1))
                histories['rating_probabilities'].append(self._decode(items, state).softmax(-1))
        result = {'rating_logits': self._decode(items, state)}
        if return_diagnostics:
            result['diagnostics'] = {name: torch.stack(values, dim=1) for name, values in histories.items()}
            result['diagnostics'].update(observed=observed, source=source)
        return result


def summarize_diagnostics(diagnostics):
    """Return only aggregate behavior, with no user IDs or individual arrays.

    Gate values are computational source weights, not identified reliability.
    Upward routing deltas use total variation across context items per port;
    downward deltas use total variation across ports per non-padding item.
    """
    observed = diagnostics['observed']
    nonempty = observed.any(dim=1)
    up, down, gates = (diagnostics[key].detach() for key in ('up', 'down', 'fidelity'))
    states, probs = (diagnostics[key].detach() for key in ('states', 'rating_probabilities'))

    def mean_or_zero(values):
        return float(values.float().mean()) if values.numel() else 0.

    rows = []
    for step in range(up.shape[1]):
        selected_gates = gates[:, step][observed]
        previous = max(step - 1, 0)
        rows.append({
            'step': step + 1,
            'up_routing_tv_from_initial': mean_or_zero((up[nonempty, step] - up[nonempty, 0]).abs().sum(-1) / 2),
            'up_routing_tv_from_previous': mean_or_zero((up[nonempty, step] - up[nonempty, previous]).abs().sum(-1) / 2),
            'down_routing_tv_from_initial': mean_or_zero((down[:, step, 1:] - down[:, 0, 1:]).abs().sum(-1) / 2),
            'down_routing_tv_from_previous': mean_or_zero((down[:, step, 1:] - down[:, previous, 1:]).abs().sum(-1) / 2),
            'source_gate_mean': mean_or_zero(selected_gates),
            'source_gate_fraction_below_005': mean_or_zero((selected_gates < .05).float()),
            'source_gate_fraction_above_095': mean_or_zero((selected_gates > .95).float()),
            'source_gate_mean_absolute_change_from_initial': mean_or_zero((gates[:, step] - gates[:, 0]).abs()[observed]),
            'state_mean_absolute_change': mean_or_zero((states[:, step + 1, 1:] - states[:, step, 1:]).abs()),
            'rating_distribution_tv_from_previous': mean_or_zero((probs[:, step + 1, 1:] - probs[:, step, 1:]).abs().sum(-1) / 2),
        })
    return {'users': int(observed.shape[0]), 'empty_context_users': int((~nonempty).sum()),
            'context_observations': int(observed.sum()), 'steps': rows}
