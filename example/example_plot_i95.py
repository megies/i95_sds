from i95_sds import I95SDSClient

client = I95SDSClient('./I95')

client.plot('BW', 'FFB1', '', 'BHZ', '2024-294', '2024-294')
client.plot('BW', 'FFB1', '', 'BHZ', '2024-294', '2024-294', type='violin')
client.plot('BW', 'FFB1', '', 'BHZ', '2024-294', '2024-294', type='line')
client.plot_availability(
    '2024-294', '2024-294', fast=False, percentage_in_label=True)
client.plot_all_data('2024-294', '2024-294')
client.plot_all_data('2024-294', '2024-294', type='violin')
