import uci_tools.vel_map

if __name__ == '__main__':
    uci_tools.vel_map.save_all_firebox_vmaps(
        res=256,
        min_cden=14.,
        bound_filter='none'
    )
