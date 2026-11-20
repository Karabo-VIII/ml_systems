# Deflated Sharpe Ratio (Lopez de Prado 2014) -- 2026-05-23

DSR = Phi( (SR - E[SR_max under null]) / SE(SR) ) where N = family size of trials.

| Strategy | N=1 DSR | N=10 DSR | N=100 DSR | N=937 DSR | SR annual |
|---|---:|---:|---:|---:|---:|
| LINK p_5 chop solo | 0.9956 | 0.8524 | 0.5362 | 0.2693 | 1.412 |
| F53 10-engine basket | 0.9970 | 0.8789 | 0.5846 | 0.3112 | 1.540 |
| F41 17-engine basket | 0.9999 | 0.9807 | 0.8668 | 0.6574 | 2.025 |
