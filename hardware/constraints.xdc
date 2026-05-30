## ============================================================================
## constraints.xdc — Timing constraints for AWS F2 (VU47P)
## Target clock: 250 MHz (4ns period)
## ============================================================================

# Primary clock
create_clock -period 4.000 -name clk_250mhz [get_ports clk]

# Clock uncertainty
set_clock_uncertainty 0.100 [get_clocks clk_250mhz]

# Input/Output delays (relative to PCIe interface)
set_input_delay -clock clk_250mhz -max 1.500 [get_ports {s_axis_tdata[*]}]
set_input_delay -clock clk_250mhz -min 0.500 [get_ports {s_axis_tdata[*]}]
set_input_delay -clock clk_250mhz -max 1.500 [get_ports s_axis_tvalid]
set_input_delay -clock clk_250mhz -min 0.500 [get_ports s_axis_tvalid]
set_input_delay -clock clk_250mhz -max 1.500 [get_ports s_axis_tlast]
set_input_delay -clock clk_250mhz -min 0.500 [get_ports s_axis_tlast]

set_output_delay -clock clk_250mhz -max 1.000 [get_ports {m_axis_tdata[*]}]
set_output_delay -clock clk_250mhz -min 0.200 [get_ports {m_axis_tdata[*]}]
set_output_delay -clock clk_250mhz -max 1.000 [get_ports m_axis_tvalid]
set_output_delay -clock clk_250mhz -min 0.200 [get_ports m_axis_tvalid]
set_output_delay -clock clk_250mhz -max 1.000 [get_ports m_axis_tlast]
set_output_delay -clock clk_250mhz -min 0.200 [get_ports m_axis_tlast]

# Configuration ports (quasi-static, relax timing)
set_input_delay -clock clk_250mhz -max 3.000 [get_ports {cfg_threshold[*]}]
set_input_delay -clock clk_250mhz -max 3.000 [get_ports {cfg_leak_factor[*]}]
set_input_delay -clock clk_250mhz -max 3.000 [get_ports cfg_start]
set_false_path -from [get_ports {cfg_threshold[*]}]
set_false_path -from [get_ports {cfg_leak_factor[*]}]

# Reset
set_input_delay -clock clk_250mhz -max 2.000 [get_ports rst_n]
set_false_path -from [get_ports rst_n]

# BRAM inference for membrane memory
set_property RAM_STYLE BLOCK [get_cells {spike_core_inst/membrane_mem_reg[*]}]

# Pipelining hint for multiplier
set_property KEEP_HIERARCHY YES [get_cells spike_core_inst]
