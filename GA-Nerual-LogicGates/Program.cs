#pragma warning disable 0169
#pragma warning disable 0414

using System;
using System.Collections.Generic;
using System.Linq;
using GANerualLogicGates.GA;
using GeneticLib.Generations;
using GeneticLib.GeneticManager;
using GeneticLib.Genome;
using GeneticLib.Genome.NeuralGenomes;
using GeneticLib.GenomeFactory;
using GeneticLib.GenomeFactory.GenomeProducer;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding.Crossover;
using GeneticLib.GenomeFactory.GenomeProducer.Breeding.Selection;
using GeneticLib.GenomeFactory.GenomeProducer.Reinsertion;
using GeneticLib.GenomeFactory.Mutation;
using GeneticLib.GenomeFactory.Mutation.NeuralMutations;
using GeneticLib.Neurology;
using GeneticLib.Randomness;
using GeneticLib.Utils.Graph;
using GeneticLib.Utils.NeuralUtils;

namespace GA_Nerual_LogicGates
{
	class Program
    {            
        int genomesCount = 50;

		float singleSynapseMutChance = 0.4f;
		float singleSynapseMutValue = 1f;
        
		float allSynapsesMutChance = 0.1f;
		float allSynapsesMutChanceEach = 0.5f;
		float allSynapsesMutValue = 0.1f;
        
        float crossoverPart = 0.80f;
        float reinsertionPart = 0.2f;

        GeneticManagerClassic geneticManager;
        public static int maxIterations = 2000;
		public static bool targetReached = false;
      
		static void Main(string[] args)
        {
			GARandomManager.Random = new RandomClassic((int)DateTime.Now.Ticks);
			var socketProxy = new SocketProxy(false);      
			var program = new Program();

			for (var i = 0; i < maxIterations; i++)
			{
				if (targetReached)
					break;

				program.Evaluate();
				var fintessSum = program.geneticManager
				                        .GenerationManager
				                        .CurrentGeneration
				                        .Genomes.Sum(x => x.Fitness);
				var best = program.BestGenome() as NeuralGenome;

				Console.WriteLine("Genes: " + best.Genes.Count());
				Console.WriteLine(String.Format(
					"{0}) Best:{1:0.00} Sum:{2:0.00}",
					i,
					best.Fitness,
					fintessSum));

				if (i % 200 == 0)
				    socketProxy.SendStrMsg(best.ToJson());

				program.Evolve();            
			}         
        }

		public Program()
		{
			var synapseTracker = new SynapseInnovNbTracker();

            var initialGenerationGenerator = new LGNeuralInitialGenerationGenerator(
                synapseTracker,
                2,
                1,
                new[] { 1 },
                () => (float)GARandomManager.Random.NextDouble(-1, 1));
            
            var selection = new EliteSelection();
            var crossover = new OnePointCrossover(true);
            var breeding = new BreedingClassic(
				crossoverPart,
                1,
				selection,
				crossover,
				InitMutations()
            );

			var reinsertion = new EliteReinsertion(reinsertionPart, 0);
			var producers = new IGenomeProducer[] { breeding, reinsertion };
			var genomeForge = new GenomeForge(producers);

			var generationManager = new GenerationManagerKeepLast();
			geneticManager = new GeneticManagerClassic(
                generationManager,
                initialGenerationGenerator,
                genomeForge,
                genomesCount
            );
   
            geneticManager.Init();
		}

		public void Evolve()
        {           
            geneticManager.Evolve();
        }

		public void Evaluate()
		{
			var genomes = geneticManager.GenerationManager
                                        .CurrentGeneration
                                        .Genomes;

            foreach (var genome in genomes)
                genome.Fitness = ComputeFitness(genome as NeuralGenome);

            var orderedGenomes = genomes.OrderByDescending(g => g.Fitness)
                                        .ToArray();

            geneticManager.GenerationManager
                          .CurrentGeneration
                          .Genomes = orderedGenomes;
		}

		private float ComputeFitness(NeuralGenome genome)
		{
			var fitness = 0d;
			for (var i = 0; i < 2; i++)
			{
				for (var j = 0; j < 2; j++)
				{
					genome.FeedNeuralNetwork(new float[] { i, j });
					var output = genome.Outputs.Select(x => x.Value).First();

					var targetValue = i ^ j;
					var delta = Math.Abs(targetValue - output);
					var gradient = (i == j && i == 1) ? 5 : 1;
					fitness -= delta * gradient;
				}
			}
			//Console.WriteLine("");
			return (float)fitness;
		}

		private MutationManager InitMutations()
		{
			var result = new MutationManager();
			//result.MutationEntries.Add(new MutationEntry(
			//	new SingleSynapseWeightMutation(() => singleSynapseMutValue),
			//	singleSynapseMutChance,
			//	EMutationType.Independent
			//));

			result.MutationEntries.Add(new MutationEntry(
				new AllSynapsesWeightMutation(
					() => allSynapsesMutValue,
					allSynapsesMutChanceEach),
				allSynapsesMutChance,
				EMutationType.Independent
			));
            
			return result;
		}

		public IGenome BestGenome()
		{
			return geneticManager.GenerationManager
								 .CurrentGeneration
								 .Genomes
								 .OrderByDescending(g => g.Fitness)
								 .First();
		}
    }
}
