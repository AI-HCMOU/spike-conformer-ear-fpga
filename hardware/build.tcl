# ==============================================================================
# build.tcl — Vivado synthesis and implementation script for SpikeConformer
# Target: AWS F2 (Xilinx VU47P, xcvu47p-fsvh2892-2-e)
#
# Usage:
#   vivado -mode batch -source hardware/build.tcl
# ==============================================================================

# Project settings
set project_name "spikeconformer_fpga"
set part         "xcvu47p-fsvh2892-2-e"
set top_module   "spike_core"
set rtl_dir      "[file dirname [info script]]/rtl"
set constr_file  "[file dirname [info script]]/constraints.xdc"
set output_dir   "[file dirname [info script]]/output"

# Create output directory
file mkdir $output_dir

# Create in-memory project
create_project -in_memory -part $part

# Add RTL sources
add_files -fileset sources_1 [glob $rtl_dir/*.v]

# Add constraints
add_files -fileset constrs_1 $constr_file

# Set top module
set_property top $top_module [current_fileset]

# Synthesis settings
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]

# Run synthesis
puts "INFO: Starting synthesis..."
synth_design -top $top_module -part $part

# Post-synthesis reports
report_timing_summary -file $output_dir/post_synth_timing.rpt
report_utilization -file $output_dir/post_synth_util.rpt
report_power -file $output_dir/post_synth_power.rpt

# Optimization
opt_design

# Placement
puts "INFO: Running placement..."
place_design
report_timing_summary -file $output_dir/post_place_timing.rpt

# Physical optimization
phys_opt_design

# Routing
puts "INFO: Running routing..."
route_design
report_timing_summary -file $output_dir/post_route_timing.rpt
report_utilization -file $output_dir/post_route_util.rpt
report_power -file $output_dir/post_route_power.rpt

# DRC
report_drc -file $output_dir/drc.rpt

# Generate bitstream
puts "INFO: Generating bitstream..."
write_bitstream -force $output_dir/${project_name}.bit

# Export hardware
write_hw_platform -fixed -force $output_dir/${project_name}.xsa

puts "INFO: Build complete. Output in $output_dir/"
puts "INFO: Bitstream: $output_dir/${project_name}.bit"
