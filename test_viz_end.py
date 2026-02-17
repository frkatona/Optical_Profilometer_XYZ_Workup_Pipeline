    # Close visualization progress bar
    if viz_pbar:
        viz_pbar.n = viz_pbar.total  # Set to 100%
        viz_pbar.close()
    
    # Add overall title with metadata (only for combined mode)
    if not separate_windows:
        filename = Path(metadata['filepath']).name
        fig.suptitle(f'Comprehensive Profilometry Analysis: {filename}\n' + 
                     f'Resolution: {data.shape[0]}x{data.shape[1]} pixels ' +
                     f'({data.shape[0]*pixel_spacing_um:.0f}×{data.shape[1]*pixel_spacing_um:.0f} µm)',
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save or show
        if output_dir:
            output_file = output_dir / f"{Path(metadata['filepath']).stem}_analysis.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_file}")
        else:
            plt.show()
        
        plt.close()
    else:
        # In separate windows mode, just show all figures
        if not output_dir:
            plt.show()
        else:
            print(f"Note: Separate windows mode with output directory saves each plot individually")
            print(f"      (Feature not yet implemented - plots will display interactively)")
