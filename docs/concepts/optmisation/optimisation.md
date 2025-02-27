Model GPT context 256
B = 64 -> crash

# Run basique
B = 2
Steps = 50
Device = cpu
Time = 100s

B = 2
Steps = 50
Device = cpu
Time = 11.45s

Time: 26.638831853866577
Tokens: 104448
Tokens/s: 3920.892649233775

B = 16 -> out of memory

# RUN 1000
step: 1000, loss: 7.5749430656433105
text: ///Je suis de la de la

step: 1500, loss: 8.27348804473877
text: ///Je suisé de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de la de///
Time: 239.3150019645691
Tokens: 8*256*1501 = 3074048
Tokens/s: 12845.195557172454
Loss: 8.27348804473877
Shard index 3

step: 2000, loss: 6.4244842529296875
text: ///Je suis, le 

step: 2500, loss: 5.838445663452148
text: ///Je suis, le étées                                            ///

step: 3000, loss: 5.503025531768799
text: ///Je suis de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la ville de la///
Time: 711.5944271087646
Tokens: 8*256*3001 = 6146048
Tokens/s: 8637.009742995919
Loss: 5.503025531768799
Shard index 6

step: 3500, loss: 5.280482292175293
text: ///Je suisées de la commune de la ville de la ville de la commune de la ville de la vie de la commune de la commune de la commune de la vie de la ville de la ville///

step: 6000, loss: 4.819955348968506
text: ///Je suis, il est un nombre de la ville de la ville de la ville de la ville de la ville de la ville.










Liens externes 
///
Time: 475.173109292984
Tokens: 8*256*6001 = 12290048
Tokens/s: 25864.35924012307
Loss: 4.819955348968506
Shard index 12