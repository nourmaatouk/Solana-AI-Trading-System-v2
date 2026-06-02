/**
 * Program IDL in camelCase format in order to be used in JS/TS.
 *
 * Note that this is only a type helper and is not the actual IDL. The original
 * IDL can be found at `target/idl/guardian.json`.
 */
export type Guardian = {
  "address": "DVQ3uopMVLznw3tLJahTkfE1VwzNSPHo1Q8tTvUbjdrG",
  "metadata": {
    "name": "guardian",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "On-chain safety gate for the Solana AI Trading System."
  },
  "instructions": [
    {
      "name": "authorizeExecution",
      "docs": [
        "The core authorisation check. Called by the backend (or any CPI caller",
        "holding the `allowed_executor` signer) immediately before invoking the",
        "Jupiter swap instruction in the same transaction. Creates a fresh",
        "`ExecutionRecord` PDA seeded by `signal_hash` — replays automatically",
        "fail because the account already exists.",
        "",
        "All four checks run before any state is written. If any one fails the",
        "transaction rolls back and the swap leg is never reached."
      ],
      "discriminator": [
        20,
        171,
        124,
        206,
        150,
        135,
        226,
        97
      ],
      "accounts": [
        {
          "name": "config",
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  103,
                  117,
                  97,
                  114,
                  100,
                  105,
                  97,
                  110,
                  45,
                  99,
                  111,
                  110,
                  102,
                  105,
                  103
                ]
              },
              {
                "kind": "account",
                "path": "config.admin",
                "account": "guardianConfig"
              }
            ]
          }
        },
        {
          "name": "executionRecord",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  101,
                  120,
                  101,
                  99,
                  117,
                  116,
                  105,
                  111,
                  110,
                  45,
                  114,
                  101,
                  99,
                  111,
                  114,
                  100
                ]
              },
              {
                "kind": "arg",
                "path": "signalHash"
              }
            ]
          }
        },
        {
          "name": "executor",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "amountLamports",
          "type": "u64"
        },
        {
          "name": "confidenceBps",
          "type": "u16"
        },
        {
          "name": "signalHash",
          "type": {
            "array": [
              "u8",
              32
            ]
          }
        }
      ]
    },
    {
      "name": "initializeConfig",
      "docs": [
        "One-time setup. Creates the `GuardianConfig` PDA and locks in the",
        "initial policy. The signer becomes the `admin` and the only account",
        "allowed to call `set_emergency_stop` or re-initialise.",
        "",
        "* `max_trade_lamports`     - Hard ceiling on `amount_lamports` passed to",
        "`authorize_execution`. Mirror of",
        "`MAX_TRADE_SIZE_SOL * LAMPORTS_PER_SOL`.",
        "* `max_daily_loss_lamports`- Currently informational (tracking lives",
        "off-chain in `daily_volume.json`); stored",
        "for future enforcement.",
        "* `confidence_threshold_bps` - 0–10_000. 6_900 ≈ off-chain",
        "`CONFIDENCE_THRESHOLD=0.69`.",
        "* `allowed_executor`       - Pubkey of the backend wallet (or Phantom",
        "key) that is permitted to call",
        "`authorize_execution`. All other signers",
        "are rejected."
      ],
      "discriminator": [
        208,
        127,
        21,
        1,
        194,
        190,
        196,
        70
      ],
      "accounts": [
        {
          "name": "config",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  103,
                  117,
                  97,
                  114,
                  100,
                  105,
                  97,
                  110,
                  45,
                  99,
                  111,
                  110,
                  102,
                  105,
                  103
                ]
              },
              {
                "kind": "account",
                "path": "admin"
              }
            ]
          }
        },
        {
          "name": "admin",
          "writable": true,
          "signer": true
        },
        {
          "name": "systemProgram",
          "address": "11111111111111111111111111111111"
        }
      ],
      "args": [
        {
          "name": "maxTradeLamports",
          "type": "u64"
        },
        {
          "name": "maxDailyLossLamports",
          "type": "u64"
        },
        {
          "name": "confidenceThresholdBps",
          "type": "u16"
        },
        {
          "name": "allowedExecutor",
          "type": "pubkey"
        }
      ]
    },
    {
      "name": "setEmergencyStop",
      "docs": [
        "Admin-only kill-switch. Flips `emergency_stop` to the supplied value.",
        "All subsequent `authorize_execution` calls fail with `EmergencyStop`",
        "until the admin flips it back."
      ],
      "discriminator": [
        185,
        103,
        202,
        188,
        168,
        230,
        104,
        177
      ],
      "accounts": [
        {
          "name": "config",
          "writable": true,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "value": [
                  103,
                  117,
                  97,
                  114,
                  100,
                  105,
                  97,
                  110,
                  45,
                  99,
                  111,
                  110,
                  102,
                  105,
                  103
                ]
              },
              {
                "kind": "account",
                "path": "admin"
              }
            ]
          }
        },
        {
          "name": "admin",
          "signer": true,
          "relations": [
            "config"
          ]
        }
      ],
      "args": [
        {
          "name": "enabled",
          "type": "bool"
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "executionRecord",
      "discriminator": [
        133,
        201,
        182,
        206,
        224,
        111,
        45,
        168
      ]
    },
    {
      "name": "guardianConfig",
      "discriminator": [
        253,
        92,
        160,
        221,
        64,
        253,
        141,
        121
      ]
    }
  ],
  "events": [
    {
      "name": "configInitialized",
      "discriminator": [
        181,
        49,
        200,
        156,
        19,
        167,
        178,
        91
      ]
    },
    {
      "name": "emergencyStopSet",
      "discriminator": [
        51,
        80,
        158,
        170,
        142,
        187,
        112,
        107
      ]
    },
    {
      "name": "executionAuthorised",
      "discriminator": [
        187,
        196,
        195,
        196,
        113,
        35,
        96,
        76
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "emergencyStop",
      "msg": "Emergency stop is active."
    },
    {
      "code": 6001,
      "name": "tradeSizeExceeded",
      "msg": "Trade amount exceeds the configured maximum."
    },
    {
      "code": 6002,
      "name": "lowConfidence",
      "msg": "Confidence is below the configured threshold."
    },
    {
      "code": 6003,
      "name": "notAdmin",
      "msg": "Only the admin can call this instruction."
    },
    {
      "code": 6004,
      "name": "notExecutor",
      "msg": "Only the configured executor can authorize a trade."
    },
    {
      "code": 6005,
      "name": "invalidAmount",
      "msg": "Amount must be a positive non-zero value."
    },
    {
      "code": 6006,
      "name": "invalidBps",
      "msg": "Basis-point value must be in [0, 10_000]."
    }
  ],
  "types": [
    {
      "name": "configInitialized",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "admin",
            "type": "pubkey"
          },
          {
            "name": "allowedExecutor",
            "type": "pubkey"
          },
          {
            "name": "maxTradeLamports",
            "type": "u64"
          },
          {
            "name": "confidenceThresholdBps",
            "type": "u16"
          }
        ]
      }
    },
    {
      "name": "emergencyStopSet",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "enabled",
            "type": "bool"
          }
        ]
      }
    },
    {
      "name": "executionAuthorised",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "executor",
            "type": "pubkey"
          },
          {
            "name": "amountLamports",
            "type": "u64"
          },
          {
            "name": "confidenceBps",
            "type": "u16"
          },
          {
            "name": "signalHash",
            "type": {
              "array": [
                "u8",
                32
              ]
            }
          }
        ]
      }
    },
    {
      "name": "executionRecord",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "signalHash",
            "type": {
              "array": [
                "u8",
                32
              ]
            }
          },
          {
            "name": "executor",
            "type": "pubkey"
          },
          {
            "name": "amountLamports",
            "type": "u64"
          },
          {
            "name": "confidenceBps",
            "type": "u16"
          },
          {
            "name": "executedAt",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "guardianConfig",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "admin",
            "type": "pubkey"
          },
          {
            "name": "allowedExecutor",
            "type": "pubkey"
          },
          {
            "name": "maxTradeLamports",
            "type": "u64"
          },
          {
            "name": "maxDailyLossLamports",
            "type": "u64"
          },
          {
            "name": "confidenceThresholdBps",
            "type": "u16"
          },
          {
            "name": "emergencyStop",
            "type": "bool"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    }
  ]
};
