// Copyright (C) 2024 Mengling LIU, The Hong Kong Polytechnic University
//
// Apache License 2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// MIT License
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the “Software”),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::sync::Arc;

use crate::circuit::types::field::SMALL_FIELD_SIZE_IN_BITS;
use crate::console::lookup_table::LookupTable;
use snarkvm_console::prelude::*;
use snarkvm_console::types::Field;
use snarkvm_console_network::Network;

#[derive(Clone)]
pub struct Table<N: Network> {
    tables: Arc<Vec<LookupTable<N>>>,
}

impl<N: Network> Table<N> {
    /// Initializes a new instance of TableExample
    pub fn setup() -> Result<Self> {
        let mut tables = vec![];
        let mut lookup_table = LookupTable::default();
        let range: u16 = 65535;
        // table 0 store  all <padded_f, one> -> padded_f
        // padded_f equals f inserted zero where f \in [0,2^16-1]
        // for example when f = i = 1010
        // padded_f = 001000001000
        for i in 0..=range {
            let key_0 = Field::<N>::from_u16(i);
            let key_1 = Field::<N>::one();
            let mut key_0_bits = key_0.to_bits_le();
            key_0_bits.truncate(SMALL_FIELD_SIZE_IN_BITS);
            let val_bits = key_0_bits
                .iter()
                .flat_map(|value| vec![value.clone(), false, false])
                .collect::<Vec<_>>();
            let val = Field::<N>::from_bits_le(&val_bits).unwrap();
            let lookup_value = [val, key_1];
            lookup_table.fill(lookup_value, val);
        }
        tables.push(lookup_table);

        // table 1 stores all values in [-2^16 + 1, 2^16 - 1]
        let mut lookup_table = LookupTable::default();
        let range: u16 = 65535;
        for i in 0..=range {
            let key_0 = Field::<N>::from_u16(i);
            let key_1 = Field::<N>::one();
            let lookup_value = [key_0, key_1];
            lookup_table.fill(lookup_value, key_0);
            let lookup_value = [-key_0, key_1];
            lookup_table.fill(lookup_value, -key_0);
        }
        tables.push(lookup_table);

        // table 2 stores  all <f, one> -> padded_f
        // padded_f equals f inserted zero where f \in [0,2^16-1]
        let mut lookup_table = LookupTable::default();
        let range: u16 = 65535;
        for i in 0..=range {
            let key_0 = Field::<N>::from_u16(i);
            let key_1 = Field::<N>::one();
            let mut key_0_bits = key_0.to_bits_le();
            key_0_bits.truncate(SMALL_FIELD_SIZE_IN_BITS);
            let val_bits = key_0_bits
                .iter()
                .flat_map(|value| vec![value.clone(), false, false])
                .collect::<Vec<_>>();
            let val = Field::<N>::from_bits_le(&val_bits).unwrap();
            let lookup_value = [key_0, key_1];
            lookup_table.fill(lookup_value, val);
        }
        tables.push(lookup_table);

        // table 3 stores all values in [0, 2^16 - 1]
        let mut lookup_table = LookupTable::default();
        let range: u16 = 65535;
        for i in 0..=range {
            let key_0 = Field::<N>::from_u16(i);
            let key_1 = Field::<N>::one();
            let lookup_value = [key_0, key_1];
            lookup_table.fill(lookup_value, key_0);
        }
        tables.push(lookup_table);

        Ok(Self {
            tables: Arc::new(tables),
        })
    }

    /// Returns the tables
    pub fn tables(&self) -> &Arc<Vec<LookupTable<N>>> {
        &self.tables
    }
}
