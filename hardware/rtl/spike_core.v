// ===========================================================================
// spike_core.v — SNN Inference Engine for FPGA
// Target: AWS F2 (Xilinx VU47P), 250 MHz clock
// Fixed-point: 16-bit (Q8.8)
// ===========================================================================
//
// This module implements a pipelined LIF neuron array for processing one
// layer of the Spiking Conformer network. Multiple instances are chained
// for multi-layer inference.
//
// Interface:
//   - AXI-Stream input: spike trains from previous layer
//   - AXI-Stream output: spike trains to next layer
//   - AXI-Lite control: configuration registers
//
// ===========================================================================

`timescale 1ns / 1ps

module spike_core #(
    parameter NUM_NEURONS    = 384,
    parameter DATA_WIDTH     = 16,     // Q8.8 fixed-point
    parameter FRAC_BITS      = 8,
    parameter TIMESTEPS      = 48,
    parameter MEM_ADDR_WIDTH = 10
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // AXI-Stream input (incoming spikes weighted by synaptic weights)
    input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                    s_axis_tvalid,
    output wire                    s_axis_tready,
    input  wire                    s_axis_tlast,

    // AXI-Stream output (output spikes)
    output reg  [DATA_WIDTH-1:0]   m_axis_tdata,
    output reg                     m_axis_tvalid,
    input  wire                    m_axis_tready,
    output reg                     m_axis_tlast,

    // Configuration (AXI-Lite mapped)
    input  wire [DATA_WIDTH-1:0]   cfg_threshold,   // Firing threshold (Q8.8)
    input  wire [DATA_WIDTH-1:0]   cfg_leak_factor, // Leak factor beta (Q8.8)
    input  wire                    cfg_start,       // Start inference pulse
    output wire                    cfg_done,        // Inference complete

    // Timestep counter
    output wire [7:0]              current_timestep
);

    // -----------------------------------------------------------------------
    // Internal signals
    // -----------------------------------------------------------------------
    localparam IDLE      = 3'd0;
    localparam PROCESS   = 3'd1;
    localparam SPIKE_OUT = 3'd2;
    localparam DONE      = 3'd3;

    reg [2:0] state, next_state;
    reg [7:0] t_count;
    reg [MEM_ADDR_WIDTH-1:0] neuron_idx;

    // Membrane potential BRAM
    reg  [DATA_WIDTH-1:0] membrane_mem [0:NUM_NEURONS-1];
    reg  [DATA_WIDTH-1:0] membrane_rd;
    wire [DATA_WIDTH-1:0] membrane_new;
    wire                   spike;

    // Pipeline registers
    reg [DATA_WIDTH-1:0] input_current;
    reg                   pipe_valid;

    // -----------------------------------------------------------------------
    // State Machine
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE: begin
                if (cfg_start)
                    next_state = PROCESS;
            end
            PROCESS: begin
                if (neuron_idx == NUM_NEURONS - 1 && pipe_valid)
                    next_state = SPIKE_OUT;
            end
            SPIKE_OUT: begin
                if (neuron_idx == NUM_NEURONS - 1 && m_axis_tready) begin
                    if (t_count == TIMESTEPS - 1)
                        next_state = DONE;
                    else
                        next_state = PROCESS;
                end
            end
            DONE: begin
                next_state = IDLE;
            end
        endcase
    end

    // -----------------------------------------------------------------------
    // Timestep counter
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            t_count <= 8'd0;
        else if (state == IDLE)
            t_count <= 8'd0;
        else if (state == SPIKE_OUT && neuron_idx == NUM_NEURONS - 1 && m_axis_tready)
            t_count <= t_count + 1;
    end

    assign current_timestep = t_count;

    // -----------------------------------------------------------------------
    // Neuron index counter
    // -----------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            neuron_idx <= 0;
        else if (state == IDLE)
            neuron_idx <= 0;
        else if (state == PROCESS && s_axis_tvalid && s_axis_tready)
            neuron_idx <= neuron_idx + 1;
        else if (state == SPIKE_OUT && m_axis_tready)
            neuron_idx <= (neuron_idx == NUM_NEURONS - 1) ? 0 : neuron_idx + 1;
    end

    // -----------------------------------------------------------------------
    // LIF Neuron: membrane_new = beta * V_old + I_in
    // spike = (membrane_new >= threshold)
    // V_out = spike ? (membrane_new - threshold) : membrane_new
    // -----------------------------------------------------------------------

    wire [2*DATA_WIDTH-1:0] leak_product;
    assign leak_product = cfg_leak_factor * membrane_rd;

    wire [DATA_WIDTH-1:0] leaked_membrane;
    assign leaked_membrane = leak_product[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS];

    assign membrane_new = leaked_membrane + input_current;
    assign spike = (membrane_new >= cfg_threshold);

    wire [DATA_WIDTH-1:0] membrane_after_spike;
    assign membrane_after_spike = spike ? (membrane_new - cfg_threshold) : membrane_new;

    // -----------------------------------------------------------------------
    // Membrane BRAM read/write
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (state == PROCESS && pipe_valid) begin
            membrane_mem[neuron_idx] <= membrane_after_spike;
        end
    end

    always @(posedge clk) begin
        membrane_rd <= membrane_mem[neuron_idx];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_current <= 0;
            pipe_valid <= 0;
        end else if (state == PROCESS && s_axis_tvalid) begin
            input_current <= s_axis_tdata;
            pipe_valid <= 1;
        end else begin
            pipe_valid <= 0;
        end
    end

    // -----------------------------------------------------------------------
    // AXI-Stream handshake
    // -----------------------------------------------------------------------
    assign s_axis_tready = (state == PROCESS);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_tdata  <= 0;
            m_axis_tvalid <= 0;
            m_axis_tlast  <= 0;
        end else if (state == SPIKE_OUT) begin
            m_axis_tdata  <= spike ? {{(DATA_WIDTH-1){1'b0}}, 1'b1} : {DATA_WIDTH{1'b0}};
            m_axis_tvalid <= 1;
            m_axis_tlast  <= (neuron_idx == NUM_NEURONS - 1);
        end else begin
            m_axis_tvalid <= 0;
            m_axis_tlast  <= 0;
        end
    end

    assign cfg_done = (state == DONE);

    // -----------------------------------------------------------------------
    // Membrane initialization
    // -----------------------------------------------------------------------
    integer i;
    always @(posedge clk) begin
        if (state == IDLE && cfg_start) begin
            for (i = 0; i < NUM_NEURONS; i = i + 1) begin
                membrane_mem[i] <= 0;
            end
        end
    end

endmodule
