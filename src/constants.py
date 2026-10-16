"""Physical constants used throughout the package. All values in SI units.

Purpose is to keep constants in one place. This is for simplicity and so that each module uses the same value.
"""

# Standard gravity [m/s^2]
G0 = 9.81

# Specific gas constant for dry air [J/(kg K)]
R_AIR = 287.05

# ISA troposphere temperature lapse rate [K/m]
LAPSE_RATE = 0.0065

# ISA sea-level standard pressure [Pa]
P0_STD = 101325.0

# ISA sea-level standard temperature [K]
T0_STD = 288.15

# Upper limit of the troposphere model [m]
H_TROPOPAUSE = 11000.0
