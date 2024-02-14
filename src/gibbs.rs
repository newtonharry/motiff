use std::io::BufRead;

use bio::io::fasta::{Reader, Records};
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use rand::prelude::*;
use scc::HashMap;

const psudeo_count: f64 = 0.001;

type Sequence = Vec<u8>;
type Sequences = Vec<SeqMotif>;

struct SeqMotif {
    seq: Sequence,
    motif_start: usize,
}

#[derive(Copy, Clone)]
enum Alphabet {
    DNA,
    Protein,
}

impl Alphabet {
    fn as_str(&self) -> &'static str {
        match self {
            Alphabet::DNA => "ACGT",
            Alphabet::Protein => "ACDEFGHIKLMNPQRSTVWY",
        }
    }

    const fn index(&self, residue: char) -> usize {
        match self {
            Alphabet::DNA => match residue {
                'A' => 0,
                'C' => 1,
                'G' => 2,
                'T' => 3,
                _ => panic!("Invalid DNA residue"),
            },
            Alphabet::Protein => match residue {
                'A' => 0,
                'C' => 1,
                'D' => 2,
                'E' => 3,
                'F' => 4,
                'G' => 5,
                'H' => 6,
                'I' => 7,
                'K' => 8,
                'L' => 9,
                'M' => 10,
                'N' => 11,
                'P' => 12,
                'Q' => 13,
                'R' => 14,
                'S' => 15,
                'T' => 16,
                'V' => 17,
                'W' => 18,
                'Y' => 19,
                _ => panic!("Invalid protein residue"),
            },
        }
    }
}

enum ProfileType {
    Background,
    Binding,
}

struct Profile {
    profile_type: ProfileType,
    matrix: HashMap<char, Array1<i64>>,
    motif_length: usize,
    alphabet: Alphabet,
    total_at_each_pos: Array1<i64>,
}

impl Profile {
    fn new(profile_type: ProfileType, length: usize, alphabet: Alphabet) -> Self {
        let matrix = HashMap::with_capacity(alphabet.as_str().len());

        for c in alphabet.as_str().chars() {
            matrix.insert(c, Array::zeros(length)).unwrap();
        }

        Self {
            profile_type,
            matrix,
            motif_length: length,
            alphabet,
            total_at_each_pos: Array::zeros(length),
        }
    }

    #[inline(always)]
    fn get_ppm(&self) -> Array2<f64> {
        let mut ppm = Array::zeros((self.alphabet.as_str().len(), self.motif_length));

        self.matrix.scan(|residue, freqs| {
            for (i, score) in freqs.iter().enumerate() {
                ppm[[self.alphabet.index(*residue), i]] = ((*score as f64)
                    + (psudeo_count / self.alphabet.as_str().len() as f64))
                    / self.total_at_each_pos[i] as f64
                    + psudeo_count;
            }
        });

        ppm
    }

    #[inline(always)]
    fn observe(&mut self, residue: char, position: usize, cnt: i64) {
        self.matrix.update(&residue, |_, v| {
            if let Some(element) = v.get_mut(position) {
                *element += cnt;
            }
        });

        self.total_at_each_pos[position] += cnt;
    }

    #[inline(always)]
    fn score(&self, motif: &[u8]) -> f64 {
        let ppm = self.get_ppm();
        let mut score = 1.0;
        for (i, residue) in motif.iter().enumerate() {
            score *= ppm[match self.profile_type {
                // NOTE:  If I need to start adding more Profile Types, I will use the type state pattern instead of match statement
                ProfileType::Binding => [self.alphabet.index(*residue as char), i], // Score each residue at each position
                ProfileType::Background => [self.alphabet.index(*residue as char), 0], // Score each residue at the first position since background profile is a 1-mer
            }];
        }
        score
    }
}

trait GibbsSampling {
    fn gibbs(&mut self, motif_length: usize, alphabet: Alphabet, niters: usize);

    fn choose_left_out_sequence(&self, num_of_seqs: usize, num_gen: &mut ThreadRng) -> usize;

    fn update_profiles_with_sequence(
        &self,
        motif_seq: &SeqMotif,
        motif_length: usize,
        ppm_profile: &mut Profile,
        background_profile: &mut Profile,
        cnt: i64,
    );

    fn choose_new_start_position(
        &self,
        seq_motif: &mut SeqMotif,
        motif_length: usize,
        ppm_profile: &Profile,
        background_profile: &Profile,
        num_gen: &mut ThreadRng,
    );

    fn generate_pwm(&self, binding_profile: &Profile, background_profile: &Profile) -> Array2<f64>;
}

impl<B> GibbsSampling for Records<B>
where
    B: BufRead,
{
    #[inline(always)]
    fn gibbs(&mut self, motif_length: usize, alphabet: Alphabet, niters: usize) {
        let mut num_gen: ThreadRng = rand::thread_rng();
        let mut seqs = self
            .filter_map(Result::ok)
            .map(|s| SeqMotif {
                seq: s.seq().to_owned(),
                motif_start: num_gen.gen_range(0..s.seq().len()),
            })
            .collect::<Sequences>();
        let num_seqs = seqs.len();

        let mut binding_profile = Profile::new(ProfileType::Binding, motif_length, alphabet);
        let mut background_profile = Profile::new(ProfileType::Background, 1, alphabet);

        let mut left_out_seq = self.choose_left_out_sequence(num_seqs, &mut num_gen);

        // Construct background and binding profiles
        for seq in seqs.iter() {
            if seq.motif_start != left_out_seq {
                self.update_profiles_with_sequence(
                    seq,
                    motif_length,
                    &mut binding_profile,
                    &mut background_profile,
                    1,
                );
            }
        }

        // Update the starting position in the left out sequence with the new start position
        self.choose_new_start_position(
            &mut seqs[left_out_seq],
            motif_length,
            &binding_profile,
            &background_profile,
            &mut num_gen,
        );

        let mut old_left_out_seq = left_out_seq;

        // Begin iterating n times to refine profiles once sequence at a time
        for _ in 0..niters {
            // Choose a sequence to leave out
            old_left_out_seq = left_out_seq;
            left_out_seq = self.choose_left_out_sequence(num_seqs, &mut num_gen);

            // Update the profile by removing the counts influenced by the newly chosen left out sequence (which was previously apart of the profile)
            // and adding the counts influenced by the old left out sequence (as it was not previously apart of the profile)
            // This is done to avoid recomputing the entire profile for each iteration

            // Add the counts influenced by the old left out sequence
            self.update_profiles_with_sequence(
                &seqs[old_left_out_seq],
                motif_length,
                &mut binding_profile,
                &mut background_profile,
                1,
            );

            // Remove the counts influenced by newly chosen left out sequence
            self.update_profiles_with_sequence(
                &seqs[left_out_seq],
                motif_length,
                &mut binding_profile,
                &mut background_profile,
                -1,
            );

            // Update the starting position in the left out sequence with the new start position
            self.choose_new_start_position(
                &mut seqs[left_out_seq],
                motif_length,
                &binding_profile,
                &background_profile,
                &mut num_gen,
            );
        }

        let pwm = self.generate_pwm(&binding_profile, &background_profile);
        let motif = (0..motif_length)
            .map(|i| {
                let highest_scoring_index = pwm.slice(s![.., i]).argmax().unwrap();
                alphabet
                    .as_str()
                    .chars()
                    .nth(highest_scoring_index)
                    .unwrap()
            })
            .collect::<String>();

        println!("Most likely motif: {}", motif);
    }

    #[inline(always)]
    fn update_profiles_with_sequence(
        &self,
        motif_seq: &SeqMotif,
        motif_length: usize,
        ppm_profile: &mut Profile,
        background_profile: &mut Profile,
        cnt: i64,
    ) {
        for (i, residue) in motif_seq.seq.iter().enumerate() {
            if i >= motif_seq.motif_start && i < motif_seq.motif_start + motif_length {
                ppm_profile.observe(*residue as char, i - motif_seq.motif_start, cnt);
            } else {
                background_profile.observe(*residue as char, 0, cnt);
            }
        }
    }

    #[inline(always)]
    fn choose_new_start_position(
        &self,
        seq_motif: &mut SeqMotif,
        motif_length: usize,
        ppm_profile: &Profile,
        background_profile: &Profile,
        num_gen: &mut ThreadRng,
    ) {
        let mut start_positions: Array1<f64> = Array::zeros(seq_motif.seq.len() - motif_length);
        for i in 0..seq_motif.seq.len() - motif_length {
            start_positions[i] = ppm_profile.score(&seq_motif.seq[i..i + motif_length])
                / background_profile.score(&seq_motif.seq[i..i + motif_length]);
        }
        start_positions /= start_positions.sum();
        let mut start_pos = 0;
        let mut cum_sum = 0.0;
        let rand_num: f64 = num_gen.gen();
        for (i, prob) in start_positions.iter().enumerate() {
            cum_sum += prob;
            if cum_sum > rand_num {
                start_pos = i;
                break;
            }
        }

        seq_motif.motif_start = start_pos;
    }

    #[inline(always)]
    fn choose_left_out_sequence(&self, num_of_seqs: usize, num_gen: &mut ThreadRng) -> usize {
        num_gen.gen_range(0..num_of_seqs)
    }

    fn generate_pwm(&self, binding_profile: &Profile, background_profile: &Profile) -> Array2<f64> {
        let binding_ppm = binding_profile.get_ppm();
        let backtground_ppm = background_profile.get_ppm();
        let mut pwm = binding_ppm / backtground_ppm;
        pwm.mapv_inplace(f64::log2);
        pwm
    }
}

#[test]
fn test_gibs_from_records() {
    let reader = Reader::from_file("../mbl_seqs.fa").expect("Need to have fasta input");
    let mut seqs = reader.records();
    seqs.gibbs(12, Alphabet::Protein, 10000);
}
